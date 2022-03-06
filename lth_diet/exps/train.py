from __future__ import annotations
from dataclasses import dataclass
from libcloud.storage.types import ObjectDoesNotExistError
import os
from pathlib import Path
import shutil
import torch
from typing import List, Optional
import wandb

import yahp as hp
from composer.algorithms import AlgorithmHparams, get_algorithm_registry
from composer.callbacks import (
    CallbackHparams,
    GradMonitorHparams,
    LRMonitorHparams,
    RunDirectoryUploaderHparams,
)
from composer.core.types import Precision
from composer.datasets import DataloaderHparams
from composer.loggers import (
    FileLoggerHparams,
    LoggerCallbackHparams,
    TQDMLoggerHparams,
    WandBLoggerHparams,
)
from composer.optim import OptimizerHparams, SchedulerHparams, SGDHparams
from composer.optim.scheduler import MultiStepLRHparams
from composer.trainer import Trainer
from composer.trainer.devices import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from composer.utils import dist, ObjectStoreProvider, ObjectStoreProviderHparams, reproducibility

from lth_diet.data import DataHparams, data_registry
from lth_diet.models import ComposerClassifierHparams, model_registry
from lth_diet.utils import utils


optimizer_registry = {"sgd": SGDHparams}
scheduler_registry = {"multistep": MultiStepLRHparams}
logger_registry = {
    "file": FileLoggerHparams,
    "wandb": WandBLoggerHparams,
    "tqdm": TQDMLoggerHparams,
}
callback_registry = {
    "lr_monitor": LRMonitorHparams,
    "grad_monitor": GradMonitorHparams,
    "run_directory_uploader": RunDirectoryUploaderHparams,
}
device_registry = {"gpu": GPUDeviceHparams, "cpu": CPUDeviceHparams}
hparams_registry = {
    "model": model_registry,
    "train_data": data_registry,
    "val_data": data_registry,
    "optimizer": optimizer_registry,
    "schedulers": scheduler_registry,
    "loggers": logger_registry,
    "algorithms": get_algorithm_registry(),
    "callbacks": callback_registry,
    "device": device_registry,
}


@dataclass
class TrainExperiment(hp.Hparams):
    hparams_registry = hparams_registry
    # required parameters
    model: ComposerClassifierHparams = hp.required("Classifier hparams")
    train_data: DataHparams = hp.required("Training data hparams")
    val_data: DataHparams = hp.required("Validation data hparams")
    train_batch_size: int = hp.required("Total across devices and grad accumulations")
    val_batch_size: int = hp.required("Total across devices and grad accumulations")
    optimizer: OptimizerHparams = hp.required("Optimizer hparams")
    schedulers: List[SchedulerHparams] = hp.required("Scheduler sequence")
    max_duration: str = hp.required("Max training time string, ep=epoch, ba=batch")
    # optional parameters
    replicate: int = hp.optional("Replicate number. Default: 0", default=0)
    seed: int = hp.optional("seed = seed * (replicate + 1). Default: 1", default=1)
    algorithms: Optional[List[AlgorithmHparams]] = hp.optional("None: []", default=None)
    callbacks: List[CallbackHparams] = hp.optional("Default: []", default_factory=list)
    loggers: List[LoggerCallbackHparams] = hp.optional("Default:[tqdm]", default_factory=lambda: [TQDMLoggerHparams()])
    device: DeviceHparams = hp.optional("Default: gpu", default=GPUDeviceHparams())
    precision: Precision = hp.optional("Default: amp", default=Precision.AMP)
    dataloader: DataloaderHparams = hp.optional("Default: Mosaic defaults", default=DataloaderHparams())
    object_store: Optional[ObjectStoreProviderHparams] = hp.optional("Default: None", default=None)
    get_name: bool = hp.optional("Print name and exit. Default: False", default=False)

    @property
    def name(self) -> str:
        ignore_fields = ["val_batch_size", "replicate", "callbacks", "loggers", "device", "precision", "dataloader"]
        ignore_fields += ["object_store", "get_name"]
        name = utils.get_hparams_name(self, prefix="TrainExperiment", ignore_fields=ignore_fields)
        return name

    def validate(self) -> None:
        super().validate()
        if self.replicate < 0:
            raise ValueError(f"Replicate must be non-negative")
        if self.seed <= 0:
            raise ValueError(f"Seed must be positive")
        world_size = dist.get_world_size()
        if self.train_batch_size % world_size != 0:
            raise ValueError(f"Train batch size not divisible by number of processes")
        if self.val_batch_size % world_size != 0:
            raise ValueError(f"Val batch size not divisible by number of processes")

    def _exists_in_bucket(self, object_store: Optional[ObjectStoreProvider]) -> bool:
        if object_store is None:
            return False
        object_name = f"exps/{utils.get_hash(self.name)}/replicate_{self.replicate}/main/ckpt_final.pt"
        try:
            object_store.get_object_size(object_name)
        except ObjectDoesNotExistError:
            return False
        return True

    def run(self) -> None:
        # Abort if a completed exp exists in the bucket, else make local exp dir and save hparams
        object_store = None if self.object_store is None else self.object_store.initialize_object()
        if self._exists_in_bucket(object_store):
            return
        exp_name = utils.get_hash(self.name)
        run_name = f"{exp_name}/replicate_{self.replicate}/main"
        run_dir = Path(os.environ["COMPOSER_RUN_DIRECTORY"])
        exp_dir = run_dir / run_name
        os.makedirs(exp_dir, exist_ok=True)
        with open(exp_dir / "hparams.yaml", "w") as f:
            f.write(self.to_yaml())

        # Device
        device = self.device.initialize_object()

        # Initialize and save initial model
        seed = self.seed * (self.replicate + 1)
        reproducibility.seed_all(seed)
        model = self.model.initialize_object()
        torch.save(model.state_dict(), exp_dir / "ckpt_init.pt")

        # Load training and validation data
        reproducibility.seed_all(42)  # prevent unwanted randomness in data generation
        train_device_batch_size = self.train_batch_size // dist.get_world_size()
        train_dataloader = self.train_data.initialize_object(train_device_batch_size, self.dataloader)
        val_device_batch_size = self.val_batch_size // dist.get_world_size()
        val_dataloader = self.val_data.initialize_object(val_device_batch_size, self.dataloader)

        # Initialize optimizer and schedulers
        optimizer = self.optimizer.initialize_object(model.parameters())
        schedulers = [x.initialize_object() for x in self.schedulers]

        # Initialize algorithms and callbacks
        algorithms = [] if self.algorithms is None else [x.initialize_object() for x in self.algorithms]
        callbacks = [x.initialize_object() for x in self.callbacks]

        # Initialize Loggers
        if self.loggers and isinstance(self.loggers[0], WandBLoggerHparams):  # extra config for WandB
            if self.loggers[0].name is None:
                self.loggers[0].name = exp_name
            if self.loggers[0].group is None:
                self.loggers[0].group = exp_name
        loggers = [x.initialize_object(config=self.to_dict()) for x in self.loggers]

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
            precision=self.precision,
            seed=seed,
            loggers=loggers,
            callbacks=callbacks,
        )

        # Save WandB run id for easy access
        if self.loggers and isinstance(self.loggers[0], WandBLoggerHparams):
            with open(exp_dir / "wandb_run_id", "w") as f:
                f.write(wandb.run.id)

        # Train model
        trainer.fit()

        # Save final model
        torch.save(model.state_dict(), exp_dir / "ckpt_final.pt")

        # If object store is provided, upload files to the cloud and clean up local directory
        if object_store is not None:
            for obj in os.listdir(exp_dir):
                object_store.upload_object(exp_dir / obj, f"exps/{run_name}/{obj}")
            shutil.rmtree(run_dir / exp_name)
