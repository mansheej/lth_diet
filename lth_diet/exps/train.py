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
from composer.loggers import WandBLoggerHparams
from composer.models import ModelHparams
from composer.optim import OptimizerHparams, SchedulerHparams, SGDHparams
from composer.optim.scheduler import MultiStepLRHparams
from composer.trainer.devices import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from composer.utils.object_store import ObjectStoreProviderHparams

from lth_diet.data import DataHparams, data_registry
from lth_diet.models import model_registry


optimizer_registry = {"sgd": SGDHparams}
scheduler_registry = {"multistep": MultiStepLRHparams}
callback_registry = {
    "lr_monitor": LRMonitorHparams,
    "grad_monitor": GradMonitorHparams,
    "run_directory_uploader": RunDirectoryUploaderHparams,
}
device_registry = {"gpu": GPUDeviceHparams, "cpu": CPUDeviceHparams}

hparams_registry = {
    "model": model_registry,
    "train_data": data_registry,
    "eval_data": data_registry,
    "optimizer": optimizer_registry,
    "schedulers": scheduler_registry,
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
    eval_data: DataHparams = hp.required("Test data hparams")
    # training
    train_batch_size: int = hp.required("Total across devices and grad accumulations")
    eval_batch_size: int = hp.required("Total across devices and grad accumulations")
    max_duration: str = hp.required("Max train time string, epoch=ep, batch=ba")
    optimizer: OptimizerHparams = hp.required("Optimizer hparams")
    schedulers: List[SchedulerHparams] = hp.required("Scheduler sequence")
    logger: WandBLoggerHparams = hp.required("WandB config")
    dataloader: DataloaderHparams = hp.required("Common dataloader hparams")
    # optional parameters
    # training
    algorithms: List[AlgorithmHparams] = hp.optional("Def: []", default_factory=list)
    callbacks: List[CallbackHparams] = hp.optional("Default: []", default_factory=list)
    device: DeviceHparams = hp.optional("Device hparams, Default: gpu", default="gpu")
    grad_clip_norm: Optional[float] = hp.optional("Max grad norm", default=None)
    validate_every_n_epochs: int = hp.optional("Default: 1", default=1)
    validate_every_n_batches: int = hp.optional("Default: -1", default=-1)
    precision: Precision = hp.optional("Default: amp", default="amp")
    # load checkpoint
    load_path: Optional[str] = hp.optional("Local disk or cloud bucket", default=None)
    load_object_store: Optional[ObjectStoreProviderHparams] = hp.optional(
        "Hparams for connecting to a cloud object store", default=None
    )
    load_weights_only: bool = hp.optional("Default: True", default=False)
    load_strict_model_weights: bool = hp.optional("Default: True", default=True)
    # save checkpoint
    save_folder: Optional[str] = hp.optional("Folder rel to run dir", default=None)
    save_interval: str = hp.optional("Time string, Default: 1ep", default="1ep")

    def run(self) -> None:
        print(self)
