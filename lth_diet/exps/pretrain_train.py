from composer.algorithms import AlgorithmHparams, get_algorithm_registry
from composer.callbacks import CallbackHparams, CheckpointSaver, GradMonitorHparams, LRMonitorHparams
from composer.core.precision import Precision
from composer.datasets import DataLoaderHparams
from composer.loggers import (
    FileLoggerHparams,
    LoggerCallbackHparams,
    TQDMLoggerHparams,
    WandBLoggerHparams,
)
from composer.optim import OptimizerHparams, SchedulerHparams, SGDHparams
from composer.optim import (
    ConstantSchedulerHparams,
    LinearSchedulerHparams,
    MultiStepSchedulerHparams,
    MultiStepWithWarmupSchedulerHparams,
)
from composer.trainer import Trainer
from composer.trainer.devices import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from composer.utils import dist, ObjectStoreProviderHparams, reproducibility, run_directory
from copy import deepcopy
import dataclasses
from lth_diet.data import DataHparams, data_registry
from lth_diet.models import ClassifierHparams, model_registry
from lth_diet.utils import utils
import os
import shutil
import torch
from typing import List, Optional
import wandb
import yahp as hp


optimizer_registry = {"sgd": SGDHparams}
scheduler_registry = {
    "constant": ConstantSchedulerHparams,
    "linear": LinearSchedulerHparams,
    "multistep": MultiStepSchedulerHparams,
    "multistep_warmup": MultiStepWithWarmupSchedulerHparams,
}
hparams_registry = {
    "model": model_registry,
    "pretrain_data": data_registry,
    "train_data": data_registry,
    "optimizer": optimizer_registry,
    "pretrain_schedulers": scheduler_registry,
    "schedulers": scheduler_registry,
    "val_data": data_registry,
    "algorithms": get_algorithm_registry(),
    "callbacks": {"lr_monitor": LRMonitorHparams, "grad_monitor": GradMonitorHparams},
    "loggers": {"file": FileLoggerHparams, "wandb": WandBLoggerHparams, "tqdm": TQDMLoggerHparams},
    "device": {"cpu": CPUDeviceHparams, "gpu": GPUDeviceHparams},
}


@dataclasses.dataclass
class PretrainAndTrainExperiment(hp.Hparams):
    hparams_registry = hparams_registry
    # required fields
    model: ClassifierHparams = hp.required("Classifier hparams")
    pretrain_data: DataHparams = hp.required("Pretraining data hparams")
    train_data: DataHparams = hp.required("Training data hparams")
    pretrain_batch_size: int = hp.required("Pretrain batch size, total across devices and grad accumulations")
    train_batch_size: int = hp.required("Training batch size, total across devices and grad accumulations")
    optimizer: OptimizerHparams = hp.required("Optimizer hparams")
    pretrain_schedulers: List[SchedulerHparams] = hp.required("Pretraining scheduler sequence")
    schedulers: List[SchedulerHparams] = hp.required("Scheduler sequence")
    pretrain_duration: str = hp.required("Time string for total pretraining, ep=epoch, ba=batch")
    max_duration: str = hp.required("Time string for total training, ep=epoch, ba=batch")
    val_data: DataHparams = hp.required("Validation data hparams")
    val_batch_size: int = hp.required("Validation batch size, total across devices")
    # optional fields
    replicate: int = hp.optional("Replicate number", default=0)
    seed: int = hp.optional("seed = seed * (replicate + 1)", default=42)
    grad_clip_norm: Optional[float] = hp.optional("None => no gradient clipping", default=None)
    algorithms: Optional[List[AlgorithmHparams]] = hp.optional("None => []", default=None)
    callbacks: Optional[List[CallbackHparams]] = hp.optional("None => []", default=None)
    loggers: List[LoggerCallbackHparams] = hp.optional("(Default: []).", default_factory=list)
    dataloader: DataLoaderHparams = hp.optional("Default: composer defaults", default=DataLoaderHparams())
    device: DeviceHparams = hp.optional("Device", default=GPUDeviceHparams())
    precision: Precision = hp.optional("Numerical precision", default=Precision.AMP)
    object_store: Optional[ObjectStoreProviderHparams] = hp.optional("Optional object store", default=None)
    get_name: bool = hp.optional("Print name and exit", default=False)

    @property
    def name(self) -> str:
        ignore_fields = ["val_data", "val_batch_size", "replicate", "loggers", "dataloader", "device", "precision"]
        ignore_fields += ["object_store", "get_name"]
        name = utils.get_hparams_name(self, prefix="PretrainTrain", ignore_fields=ignore_fields)
        return name

    def _train(self, pretrain: bool) -> None:
        # Experiment name
        exp_hash = utils.get_hash(self.name)
        phase = "pretrain" if pretrain else "train"
        exp_name = f"{exp_hash}/replicate_{self.replicate}/{phase}/main"

        # If experiment completed, abort
        object_store = None if self.object_store is None else self.object_store.initialize_object()
        object_name = f"{os.environ['OBJECT_STORE_DIR']}/{exp_name}/state_final.pt"
        if utils.object_exists_in_bucket(object_name, object_store):
            print(f"{object_name} exists in bucket")
            return

        # Make new local exp dir
        exp_dir = os.path.join(run_directory.get_run_directory(), exp_name)
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
        os.makedirs(exp_dir)

        # Save hparams
        with open(os.path.join(exp_dir, "hparams.yaml"), "w") as f:
            f.write(self.to_yaml())

        # Device
        device = self.device.initialize_object()

        # Initialize and save initial model
        seed = self.seed * (self.replicate + 1)  # Adjust seed for replicate
        reproducibility.seed_all(seed)  # Seed rngs before randomly initializing the model
        model = self.model.initialize_object()
        if pretrain:  # if pretraining, we just save the randomly initialized model
            torch.save(model.state_dict(), os.path.join(exp_dir, "model_init.pt"))
        else:
            # pretrain model name will have phase = pretrain
            pretrain_model_name = f"{exp_hash}/replicate_{self.replicate}/pretrain/main/state_final.pt"
            # initial train state should be in the current experiment directory
            path_init_train_state = os.path.join(exp_dir, "state_init.pt")
            # first look for checkpoint in object store, if found download it
            object_name = f"{os.environ['OBJECT_STORE_DIR']}/{pretrain_model_name}"
            if utils.object_exists_in_bucket(object_name, object_store):
                object_store.download_object(object_name, path_init_train_state)
            else:
                # if not in object store, look for checkpoint in local file system
                path_final_pretrain_state = os.path.join(run_directory.get_run_directory(), pretrain_model_name)
                if os.path.exists(path_final_pretrain_state):
                    shutil.copy(path_final_pretrain_state, path_init_train_state)
                else:
                    raise FileNotFoundError("Pretrain final state not found.")

        # Load training and validation data
        reproducibility.seed_all(42)  # prevent unwanted randomness in data generation
        train_batch_size = self.pretrain_batch_size if pretrain else self.train_batch_size
        train_data = self.pretrain_data if pretrain else self.train_data
        train_device_batch_size = train_batch_size // dist.get_world_size()
        train_dataloader = train_data.initialize_object(
            train_device_batch_size, self.dataloader, replicate=self.replicate, object_store=object_store
        )
        val_device_batch_size = self.val_batch_size // dist.get_world_size()
        val_dataloader = self.val_data.initialize_object(
            val_device_batch_size, self.dataloader, replicate=self.replicate, object_store=object_store
        )

        # Initialize optimizer and schedulers, same optimizer hparam used by pretrain and train phases
        optimizer = deepcopy(self.optimizer).initialize_object(model.parameters())
        schedulers = self.pretrain_schedulers if pretrain else self.schedulers
        schedulers = [x.initialize_object() for x in schedulers]

        # Initialize algorithms and callbacks, deepcopy because used by pretrain and train phases
        # CheckpointSaver saves final state
        algorithms = [] if self.algorithms is None else [deepcopy(x).initialize_object() for x in self.algorithms]
        callbacks = [] if self.callbacks is None else [deepcopy(x).initialize_object() for x in self.callbacks]
        callbacks.append(
            CheckpointSaver(
                save_folder=exp_dir,
                name_format="state_final.pt",
                save_latest_format=None,
                save_interval=utils.save_final,
            )
        )

        # Configure and initialize loggers, deepcopy because used by pretrain and train phases
        loggers, config_dict, save_wandb_run_id = [], self.to_dict(), False
        for logger in self.loggers:
            logger = deepcopy(logger)
            if isinstance(logger, FileLoggerHparams):
                logger.filename = os.path.join(exp_dir, "log.txt")
                logger.flush_interval = len(train_dataloader)
            elif isinstance(logger, WandBLoggerHparams):
                logger.name = f"{exp_hash}_{self.replicate}_{phase}"
                logger.group = exp_hash
                save_wandb_run_id = True
            loggers.append(logger.initialize_object(config=config_dict))

        # Initialize trainer
        duration = self.pretrain_duration if pretrain else self.max_duration
        load_path = None if pretrain else path_init_train_state
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            max_duration=duration,
            eval_dataloader=val_dataloader,
            algorithms=algorithms,
            optimizers=optimizer,
            schedulers=schedulers,
            device=device,
            grad_clip_norm=self.grad_clip_norm,
            precision=self.precision,
            step_schedulers_every_batch=True,
            seed=seed,
            loggers=loggers,
            callbacks=callbacks,
            load_path_format=load_path,
        )

        # Save WandB run id for easy access
        if save_wandb_run_id:
            with open(os.path.join(exp_dir, "wandb_run_id.txt"), "w") as f:
                f.write(wandb.run.id)

        # Train model
        trainer.fit()

        # If object store is provided, upload files to the cloud and clean up local directory
        if object_store is not None:
            for obj in os.listdir(exp_dir):
                object_store.upload_object(
                    os.path.join(exp_dir, obj), f"{os.environ['OBJECT_STORE_DIR']}/{exp_name}/{obj}"
                )
            shutil.rmtree(exp_dir)

        return

    def run(self) -> None:
        # Assert batch sizes
        assert (
            self.pretrain_batch_size % dist.get_world_size() == 0
        ), "Pretrain batch size not divisible by number of processes"
        assert self.train_batch_size % dist.get_world_size() == 0, "Train batch size not div by number of processes"
        assert self.val_batch_size % dist.get_world_size() == 0, "Val batch size not div by number of processes"

        # Pretrain and then train
        self._train(pretrain=True)
        self._train(pretrain=False)

        return
