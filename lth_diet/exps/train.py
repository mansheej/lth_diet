from dataclasses import dataclass
from typing import List, Optional

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
from composer.models import ModelHparams
from composer.optim import OptimizerHparams, SchedulerHparams, SGDHparams
from composer.optim.scheduler import MultiStepLRHparams
from composer.trainer.devices import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from composer.utils import dist
from composer.utils.object_store import ObjectStoreProviderHparams

from lth_diet.data import DataHparams, data_registry
from lth_diet.models import model_registry


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
    # experiment
    replicate: int = hp.required("Replicate number")
    model_seed: int = hp.required("Init model with model_seed * (replicate + 1)")
    sgd_seed: int = hp.required("Init SGD noise with sgd_seed * (replicate + 1)")
    # model and data
    model: ModelHparams = hp.required("Model hparams")
    train_data: DataHparams = hp.required("Training data hparams")
    val_data: DataHparams = hp.required("Validation data hparams")
    # training
    max_duration: str = hp.required("Max training time string, ep=epoch, ba=batch")
    train_batch_size: int = hp.required("Total across devices and grad accumulations")
    val_batch_size: int = hp.required("Total across devices and grad accumulations")
    dataloader: DataloaderHparams = hp.required("Common dataloader hparams")
    optimizer: OptimizerHparams = hp.required("Optimizer hparams")
    schedulers: List[SchedulerHparams] = hp.required("Scheduler sequence")
    loggers: List[LoggerCallbackHparams] = hp.required("Loggers")
    # optional parameters
    # training
    algorithms: List[AlgorithmHparams] = hp.optional("Default:[]", default_factory=list)
    callbacks: List[CallbackHparams] = hp.optional("Default: []", default_factory=list)
    device: DeviceHparams = hp.optional("Default: gpu", default="gpu")
    precision: Precision = hp.optional("Default: amp", default="amp")
    # checkpoint
    load_object_store: Optional[ObjectStoreProviderHparams] = hp.optional(
        "Hparams for connecting to a cloud object store", default=None
    )

    def validate(self) -> None:
        super().validate()
        world_size = dist.get_world_size()
        if self.train_batch_size % world_size != 0:
            raise ValueError(
                f"Batch size ({self.train_batch_size}) not divisible by the total number of processes ({world_size})."
            )
        if self.val_batch_size % world_size != 0:
            raise ValueError(
                f"Eval batch size ({self.val_batch_size}) not divisible by the total number of processes ({world_size})."
            )

    def run(self) -> None:
        print(self)
