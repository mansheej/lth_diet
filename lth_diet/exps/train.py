from dataclasses import dataclass
from typing import List

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
from composer.optim.scheduler import ensure_warmup_last, MultiStepLRHparams
from composer.trainer import Trainer
from composer.trainer.devices import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from composer.utils import dist, reproducibility

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
        # get device
        device = self.device.initialize_object()

        # train data
        reproducibility.seed_all(42)
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
        model_seed = self.model_seed * (self.replicate + 1)
        reproducibility.seed_all(model_seed)
        model = self.model.initialize_object()

        # optimizer and scheduler
        optimizer = self.optimizer.initialize_object(model.parameters())
        steps_per_epoch = len(train_dataloader)
        samples_per_epoch = steps_per_epoch * self.train_batch_size
        schedulers = [
            x.initialize_object(
                optimizer=optimizer,
                max_training_duration=self.max_duration,
                steps_per_epoch=steps_per_epoch,
                samples_per_epoch=samples_per_epoch,
                dataset_num_tokens=None,
            )
            for x in ensure_warmup_last(self.schedulers)
        ]

        # algorithms, callbacks, and loggers
        algorithms = [x.initialize_object() for x in self.algorithms]
        callbacks = [x.initialize_object() for x in self.callbacks]
        loggers = [x.initialize_object(self.to_dict()) for x in self.loggers]

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
            seed=model_seed,
            loggers=loggers,
            callbacks=callbacks,
        )

        # train
        sgd_seed = self.sgd_seed * (self.replicate + 1)
        reproducibility.seed_all(sgd_seed)
        trainer.fit()
