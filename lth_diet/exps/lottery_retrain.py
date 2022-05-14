from __future__ import annotations

import dataclasses
import os
import shutil
from copy import deepcopy
from typing import List, Optional

import torch
import wandb
import yahp as hp
from composer.algorithms import AlgorithmHparams, get_algorithm_registry
from composer.callbacks import CallbackHparams, GradMonitorHparams, LRMonitorHparams
from composer.core.precision import Precision
from composer.core.time import Time
from composer.datasets import DataLoaderHparams
from composer.loggers import FileLoggerHparams, LoggerCallbackHparams, TQDMLoggerHparams, WandBLoggerHparams
from composer.models import ComposerClassifier
from composer.optim import (
    ConstantSchedulerHparams,
    LinearSchedulerHparams,
    MultiStepSchedulerHparams,
    MultiStepWithWarmupSchedulerHparams,
    OptimizerHparams,
    SchedulerHparams,
    SGDHparams,
)
from composer.trainer import Trainer
from composer.trainer.devices import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from composer.utils import ObjectStoreProvider, ObjectStoreProviderHparams, dist, reproducibility, run_directory
from lth_diet.data import DataHparams, data_registry
from lth_diet.exps import LotteryExperiment
from lth_diet.models import ClassifierHparams, model_registry
from lth_diet.pruning import Mask, PrunedClassifier, PruningHparams
from lth_diet.utils import utils

optimizer_registry = {"sgd": SGDHparams}
scheduler_registry = {
    "constant": ConstantSchedulerHparams,
    "linear": LinearSchedulerHparams,
    "multistep": MultiStepSchedulerHparams,
    "multistep_warmup": MultiStepWithWarmupSchedulerHparams,
}
hparams_registry = {
    "model": model_registry,
    "train_data": data_registry,
    "optimizer": optimizer_registry,
    "schedulers": scheduler_registry,
    "val_data": data_registry,
    "pretrain_data": data_registry,
    "pretrain_optimizer": optimizer_registry,
    "pretrain_schedulers": scheduler_registry,
    "algorithms": get_algorithm_registry(),
    "callbacks": {"lr_monitor": LRMonitorHparams, "grad_monitor": GradMonitorHparams},
    "loggers": {"file": FileLoggerHparams, "wandb": WandBLoggerHparams, "tqdm": TQDMLoggerHparams},
    "device": {"cpu": CPUDeviceHparams, "gpu": GPUDeviceHparams},
}


@dataclasses.dataclass
class LotteryRetrainExperiment(hp.Hparams):
    hparams_registry = hparams_registry
    # required fields
    model: ClassifierHparams = hp.required("Classifier hparams")
    load_exp: LotteryExperiment = hp.required("Hparams for lottery experiment to load model from")
    load_replicate: int = hp.required("Replicate for lottery experiment to load model from")
    train_data: DataHparams = hp.required("Training data hparams")
    train_batch_size: int = hp.required("Training batch size, total across devices and grad accumulations")
    optimizer: OptimizerHparams = hp.required("Optimizer hparams")
    schedulers: List[SchedulerHparams] = hp.required("Scheduler sequence")
    max_duration: str = hp.required("Time string for total training, ep=epoch, ba=batch")
    val_data: DataHparams = hp.required("Validation data hparams")
    val_batch_size: int = hp.required("Validation batch size, total across devices")
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
        ignore_fields = ["val_data", "val_batch_size", "replicate", "loggers", "dataloader", "device"]
        ignore_fields += ["precision", "object_store", "get_name", "load_exp", "load_replicate"]
        name = utils.get_hparams_name(self, prefix="LotteryExperiment", ignore_fields=ignore_fields)
        return name

    def run(self) -> None:
        # distributed training not supported and object_store required
        assert dist.get_world_size() == 1, "Distributed training not currently supported"
        assert self.object_store is not None, "Object store is currently required"

        # object store
        object_store = self.object_store.initialize_object()

        # parent LotteryExperiment (error if parent doesn't exist)
        parent_exp_hash = utils.get_hash(self.load_exp.name)
        parent_location = f"{parent_exp_hash}/replicate_{self.load_replicate}/level_0/main"
        parent_object_name = utils.get_object_name(parent_location, "model_init.pt")
        if not utils.object_exists_in_bucket(parent_object_name, object_store):
            raise ValueError(f"Expected parent does not exist at {parent_object_name}")

        # child experiment
        exp_hash = f"{utils.get_hash(self.name)}"
        branch = f"{exp_hash}/replicate_{self.replicate}/main"
        location = f"{parent_exp_hash}/replicate_{self.load_replicate}/level_0/{branch}"

        # if experiment completed, abort
        if utils.object_exists_in_bucket(utils.get_object_name(location, "model_final.pt"), object_store):
            print(f"{utils.get_object_name(location, 'model_final.pt')} exists in bucket")
            return

        # print and make local directory, save hparams
        print("-" * 80 + f"\nRetrain from {parent_object_name}\n" + "-" * 80)
        if os.path.exists(utils.get_local_dir(location)):  # setup local exp dir
            shutil.rmtree(utils.get_local_dir(location))
        os.makedirs(utils.get_local_dir(location))
        with open(utils.get_local_path(location, "hparams.yaml"), "w") as f:  # save hparams
            f.write(self.to_yaml())

        # download parent model and mask
        object_store.download_object(parent_object_name, utils.get_local_path(location, "model_init.pt"))
        object_store.download_object(
            utils.get_object_name(parent_location, "mask.pt"), utils.get_local_path(location, "mask.pt")
        )

        # initialize model
        seed = self.seed * (self.replicate + 1)
        state_dict = torch.load(utils.get_local_path(location, "model_init.pt"))
        model = self.model.initialize_object()
        model.module.load_state_dict(state_dict)
        mask = torch.load(utils.get_local_path(location, "mask.pt"))
        model = PrunedClassifier(model, mask)

        # device
        device = self.device.initialize_object()

        # load training and validation data
        reproducibility.seed_all(42)  # prevent unwanted randomness in data generation
        train_dataloader = self.train_data.initialize_object(
            self.train_batch_size, self.dataloader, replicate=self.replicate, object_store=object_store
        )
        val_dataloader = self.val_data.initialize_object(
            self.val_batch_size, self.dataloader, replicate=self.replicate, object_store=object_store
        )

        # Initialize optimizer and schedulers
        optimizer = self.optimizer.initialize_object(model.parameters())
        schedulers = [x.initialize_object() for x in self.schedulers]

        # Initialize algorithms and callbacks, deepcopy because used by pretrain and IMP levels
        algorithms = [] if self.algorithms is None else [x.initialize_object() for x in self.algorithms]
        callbacks = [] if self.callbacks is None else [x.initialize_object() for x in self.callbacks]

        # Configure and initialize loggers, deepcopy because used by pretrain and train phases
        loggers, config_dict, save_wandb_run_id = [], self.to_dict(), False
        for logger in self.loggers:
            if isinstance(logger, FileLoggerHparams):
                logger.filename = utils.get_local_path(location, "log.txt")
                logger.flush_interval = len(train_dataloader)
            elif isinstance(logger, WandBLoggerHparams):
                logger.name = f"{parent_exp_hash}_{self.load_replicate}_{exp_hash}_{self.replicate}"
                logger.group = f"{parent_exp_hash}_{self.load_replicate}_{exp_hash}"
                save_wandb_run_id = True
            loggers.append(logger.initialize_object(config=config_dict))

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
            with open(utils.get_local_path(location, "wandb_run_id.txt"), "w") as f:
                f.write(wandb.run.id)

        # Fastforward to rewinding step
        rewind_time, steps = Time.from_timestring(self.load_exp.rewinding_steps), 0
        while trainer.state.timer < rewind_time:
            trainer.state.timer.on_batch_complete(self.train_batch_size)
            steps += 1
            if steps % trainer.state.steps_per_epoch == 0:
                trainer.state.timer.on_epoch_complete()

        # Train model
        trainer.fit()

        # Save final model
        torch.save(model.module.state_dict(), utils.get_local_path(location, "model_final.pt"))

        # Upload files to the cloud and clean up local directory
        for obj in os.listdir(utils.get_local_dir(location)):
            object_store.upload_object(utils.get_local_path(location, obj), utils.get_object_name(location, obj))
        shutil.rmtree(utils.get_local_dir(location))
