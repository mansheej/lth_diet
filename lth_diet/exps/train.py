from dataclasses import dataclass
import dotenv
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
from composer.optim import OptimizerHparams, SchedulerHparams, SGDHparams
from composer.optim.scheduler import MultiStepLRHparams
from composer.trainer import Trainer
from composer.trainer.devices import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from composer.utils import dist, reproducibility

from lth_diet.data import DataHparams, data_registry
from lth_diet.models import ClassifierHparams, model_registry


dotenv.load_dotenv()

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
    seed: int = hp.required("seed = seed * (replicate + 1)")
    # model and data
    model: ClassifierHparams = hp.required("Model hparams")
    train_data: DataHparams = hp.required("Training data hparams")
    val_data: DataHparams = hp.required("Validation data hparams")
    # training
    max_duration: str = hp.required("Max training time string, ep=epoch, ba=batch")
    train_batch_size: int = hp.required("Total across devices and grad accumulations")
    val_batch_size: int = hp.required("Total across devices and grad accumulations")
    optimizer: OptimizerHparams = hp.required("Optimizer hparams")
    schedulers: List[SchedulerHparams] = hp.required("Scheduler sequence")
    dataloader: DataloaderHparams = hp.required("Common dataloader hparams")
    # optional parameters
    # training
    algorithms: List[AlgorithmHparams] = hp.optional("Default:[]", default_factory=list)
    callbacks: List[CallbackHparams] = hp.optional("Default: []", default_factory=list)
    loggers: List[LoggerCallbackHparams] = hp.optional(
        "Default: [tqdm]", default_factory=lambda: [TQDMLoggerHparams()]
    )
    device: DeviceHparams = hp.optional("Default: gpu", default=GPUDeviceHparams())
    precision: Precision = hp.optional("Default: amp", default=Precision.AMP)
    # checkpoints
    save_interval: Optional[str] = hp.optional("Default: None (Nba=1ep)", default=None)

    def validate(self) -> None:
        super().validate()
        if self.replicate < 0:
            raise ValueError(f"Replicate must be positive")
        if self.seed <= 0:
            raise ValueError(f"Seed must be non-negative")
        world_size = dist.get_world_size()
        if self.train_batch_size % world_size != 0:
            raise ValueError(f"Train batch size not divisible by number of processes")
        if self.val_batch_size % world_size != 0:
            raise ValueError(f"Val batch size not divisible by number of processes")

    def run(self) -> None:
        # get device
        device = self.device.initialize_object()

        # train data
        reproducibility.seed_all(42)  # prevent unwanted randomness in data generation
        train_device_batch_size = self.train_batch_size // dist.get_world_size()
        train_dataloader = self.train_data.initialize_object(
            train_device_batch_size, self.dataloader
        )
        # validation data
        val_device_batch_size = self.val_batch_size // dist.get_world_size()
        val_dataloader = self.val_data.initialize_object(
            val_device_batch_size, self.dataloader
        )

        # model
        seed = self.seed * (self.replicate + 1)
        reproducibility.seed_all(seed)
        model = self.model.initialize_object()

        # optimizer and scheduler
        optimizer = self.optimizer.initialize_object(model.parameters())
        schedulers = [x.initialize_object() for x in self.schedulers]

        # algorithms, callbacks, and loggers
        algorithms = [x.initialize_object() for x in self.algorithms]
        callbacks = [x.initialize_object() for x in self.callbacks]
        loggers = [x.initialize_object(config=self.to_dict()) for x in self.loggers]

        # checkpointing
        save_interval = self.save_interval
        if save_interval is None:
            save_interval = f"{len(train_dataloader)}ba"

        # trainer
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
            save_folder="temp",
            save_interval=save_interval,
        )

        # train
        trainer.fit()
