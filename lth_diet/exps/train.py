from __future__ import annotations
from composer.algorithms import AlgorithmHparams, get_algorithm_registry
from composer.callbacks import CallbackHparams, GradMonitorHparams, LRMonitorHparams
from composer.core.precision import Precision
from composer.datasets import DataLoaderHparams
from composer.loggers import FileLoggerHparams, LoggerCallbackHparams, TQDMLoggerHparams, WandBLoggerHparams
from composer.optim import (
    OptimizerHparams,
    MultiStepSchedulerHparams,
    MultiStepWithWarmupSchedulerHparams,
    SchedulerHparams,
    SGDHparams,
)
from composer.trainer import Trainer
from composer.trainer.devices import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from composer.utils import dist, ObjectStoreProviderHparams, reproducibility, run_directory
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


hparams_registry = {
    "model": model_registry,
    "train_data": data_registry,
    "val_data": data_registry,
    "optimizer": {"sgd": SGDHparams},
    "schedulers": {"multistep": MultiStepSchedulerHparams, "multistep_warmup": MultiStepWithWarmupSchedulerHparams},
    "algorithms": get_algorithm_registry(),
    "callbacks": {"lr_monitor": LRMonitorHparams, "grad_monitor": GradMonitorHparams},
    "loggers": {"file": FileLoggerHparams, "wandb": WandBLoggerHparams, "tqdm": TQDMLoggerHparams},
    "device": {"cpu": CPUDeviceHparams, "gpu": GPUDeviceHparams},
}


@dataclasses.dataclass
class TrainExperiment(hp.Hparams):
    hparams_registry = hparams_registry
    # required fields
    model: ClassifierHparams = hp.required("Classifier hparams")
    train_data: DataHparams = hp.required("Training data hparams")
    train_batch_size: int = hp.required("Total across devices and grad accumulations")
    optimizer: OptimizerHparams = hp.required("Optimizer hparams")
    schedulers: List[SchedulerHparams] = hp.required("Scheduler sequence")
    max_duration: str = hp.required("Max training time string, ep=epoch, ba=batch")
    val_data: DataHparams = hp.required("Validation data hparams")
    val_batch_size: int = hp.required("Total across devices and grad accumulations")
    # optional fields
    replicate: int = hp.optional("Replicate number", default=0)
    seed: int = hp.optional("seed = seed * (replicate + 1)", default=42)
    grad_clip_norm: Optional[float] = hp.optional("None => no gradient clipping", default=None)
    algorithms: Optional[List[AlgorithmHparams]] = hp.optional("None => []", default=None)
    callbacks: List[CallbackHparams] = hp.optional("(Default: []).", default_factory=list)
    loggers: List[LoggerCallbackHparams] = hp.optional("(Default: []).", default_factory=list)
    dataloader: DataLoaderHparams = hp.optional("Default: composer defaults", default=DataLoaderHparams())
    device: DeviceHparams = hp.optional("Device", default=GPUDeviceHparams())
    precision: Precision = hp.optional("Numerical precision", default=Precision.AMP)
    object_store: Optional[ObjectStoreProviderHparams] = hp.optional("Optional object store", default=None)
    get_name: bool = hp.optional("Print name and exit", default=False)

    @property
    def name(self) -> str:
        ignore_fields = ["val_data", "val_batch_size", "replicate", "loggers", "dataloader", "device", "object_store"]
        ignore_fields += ["get_name"]
        name = utils.get_hparams_name(self, prefix="Train", ignore_fields=ignore_fields)
        return name

    def run(self) -> None:
        # Assert batch sizes
        assert self.train_batch_size % dist.get_world_size() == 0, "Train batch size not div by number of processes"
        assert self.val_batch_size % dist.get_world_size() == 0, "Val batch size not div by number of processes"

        # Name experiment
        exp_hash = utils.get_hash(self.name)
        exp_name = f"{exp_hash}/replicate_{self.replicate}/main"

        # If experiment completed, abort
        object_name = f"{os.environ['OBJECT_STORE_DIR']}/{exp_name}/model_final.pt"
        object_store = None if self.object_store is None else self.object_store.initialize_object()
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
        torch.save(model.state_dict(), os.path.join(exp_dir, "model_init.pt"))

        # Load training and validation data
        reproducibility.seed_all(42)  # prevent unwanted randomness in data generation
        train_device_batch_size = self.train_batch_size // dist.get_world_size()
        train_dataloader = self.train_data.initialize_object(
            train_device_batch_size, self.dataloader, replicate=self.replicate, object_store=object_store
        )
        val_device_batch_size = self.val_batch_size // dist.get_world_size()
        val_dataloader = self.val_data.initialize_object(
            val_device_batch_size, self.dataloader, replicate=self.replicate, object_store=object_store
        )

        # Initialize optimizer and schedulers
        optimizer = self.optimizer.initialize_object(model.parameters())
        schedulers = [x.initialize_object() for x in self.schedulers]

        # Initialize algorithms and callbacks
        algorithms = [] if self.algorithms is None else [x.initialize_object() for x in self.algorithms]
        callbacks = [x.initialize_object() for x in self.callbacks]

        # Configure and initialize loggers
        save_wandb_run_id = False
        for logger in self.loggers:
            if isinstance(logger, FileLoggerHparams):
                logger.filename = os.path.join(exp_dir, logger.filename)
                logger.flush_interval = len(train_dataloader)
            elif isinstance(logger, WandBLoggerHparams):
                logger.name = f"{exp_hash}_{self.replicate}"
                logger.group = exp_hash
                save_wandb_run_id = True
        config_dict = self.to_dict()
        loggers = [x.initialize_object(config=config_dict) for x in self.loggers]

        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            max_duration=self.max_duration,
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
        )

        # Save WandB run id for easy access
        if save_wandb_run_id:
            with open(os.path.join(exp_dir, "wandb_run_id.txt"), "w") as f:
                f.write(wandb.run.id)

        # Train model
        trainer.fit()

        # Save final model
        torch.save(model.state_dict(), os.path.join(exp_dir, "model_final.pt"))

        # If object store is provided, upload files to the cloud and clean up local directory
        if object_store is not None:
            for obj in os.listdir(exp_dir):
                object_store.upload_object(
                    os.path.join(exp_dir, obj), f"{os.environ['OBJECT_STORE_DIR']}/{exp_name}/{obj}"
                )
            shutil.rmtree(exp_dir)

        return
