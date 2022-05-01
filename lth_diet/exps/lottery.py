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
class LotteryExperiment(hp.Hparams):
    hparams_registry = hparams_registry
    # required fields
    model: ClassifierHparams = hp.required("Classifier hparams")
    train_data: DataHparams = hp.required("Training data hparams")
    train_batch_size: int = hp.required("Training batch size, total across devices and grad accumulations")
    optimizer: OptimizerHparams = hp.required("Optimizer hparams")
    schedulers: List[SchedulerHparams] = hp.required("Scheduler sequence")
    max_duration: str = hp.required("Time string for total training, ep=epoch, ba=batch")
    val_data: DataHparams = hp.required("Validation data hparams")
    val_batch_size: int = hp.required("Validation batch size, total across devices")
    # optional fields
    replicate: int = hp.optional("Replicate number", default=0)
    seed: int = hp.optional("seed = seed * (replicate + 1)", default=42)
    levels: int = hp.optional("Number of IMP levels", default=20)
    rewinding_steps: str = hp.optional("Rewind to this step (in batches) during IMP", default="0ba")
    pruning: PruningHparams = hp.optional("Default: pruning_fraction = 0.2", default=PruningHparams())
    pretrain_data: Optional[DataHparams] = hp.optional("None => train_data", default=None)
    pretrain_batch_size: Optional[int] = hp.optional("None => train_batch_size", default=None)
    pretrain_optimizer: Optional[OptimizerHparams] = hp.optional("None => optimizer", default=None)
    pretrain_schedulers: Optional[List[SchedulerHparams]] = hp.optional("None => schedulers", default=None)
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
        ignore_fields = ["val_data", "val_batch_size", "replicate", "levels", "loggers", "dataloader", "device"]
        ignore_fields += ["precision", "object_store", "get_name"]
        name = utils.get_hparams_name(self, prefix="Lottery", ignore_fields=ignore_fields)
        return name

    def run(self) -> None:
        # distributed training not supported and object_store required
        assert dist.get_world_size() == 1, "Distributed training not currently supported"
        assert self.object_store is not None, "Object store is currently required"

        # global
        exp_hash = utils.get_hash(self.name)
        object_store = self.object_store.initialize_object()

        # setup weights for IMP
        if self.rewinding_steps != "0ba":
            self._pretrain(exp_hash, object_store)
        self._establish_initial_weights(exp_hash, object_store)

        # train IMP levels
        for level in range(self.levels + 1):
            self._prune_level(level, exp_hash, object_store)
            self._train_level(level, exp_hash, object_store)

    def _train(
        self,
        exp_hash: str,
        level: str,
        model: ComposerClassifier,
        train_data: DataHparams,
        train_batch_size: int,
        optimizer: OptimizerHparams,
        schedulers: List[SchedulerHparams],
        duration: str,
        seed: int,
        object_store: ObjectStoreProvider,
    ) -> None:
        # location
        location = f"{exp_hash}/replicate_{self.replicate}/{level}/main"

        # device
        device = self.device.initialize_object()

        # load training and validation data
        reproducibility.seed_all(42)  # prevent unwanted randomness in data generation
        train_dataloader = train_data.initialize_object(
            train_batch_size, self.dataloader, replicate=self.replicate, object_store=object_store
        )
        val_dataloader = self.val_data.initialize_object(
            self.val_batch_size, self.dataloader, replicate=self.replicate, object_store=object_store
        )

        # Initialize optimizer and schedulers
        optimizer = optimizer.initialize_object(model.parameters())
        schedulers = [x.initialize_object() for x in schedulers]

        # Initialize algorithms and callbacks, deepcopy because used by pretrain and IMP levels
        algorithms = [] if self.algorithms is None else [deepcopy(x).initialize_object() for x in self.algorithms]
        callbacks = [] if self.callbacks is None else [deepcopy(x).initialize_object() for x in self.callbacks]

        # Configure and initialize loggers, deepcopy because used by pretrain and train phases
        loggers, config_dict, save_wandb_run_id = [], self.to_dict(), False
        for logger in self.loggers:
            logger = deepcopy(logger)
            if isinstance(logger, FileLoggerHparams):
                logger.filename = utils.get_local_path(location, "log.txt")
                logger.flush_interval = len(train_dataloader)
            elif isinstance(logger, WandBLoggerHparams):
                logger.name = f"{exp_hash}_{self.replicate}_{level}"
                logger.group = exp_hash
                save_wandb_run_id = True
            loggers.append(logger.initialize_object(config=config_dict))

        # Initialize trainer
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
        )

        # Save WandB run id for easy access
        if save_wandb_run_id:
            with open(utils.get_local_path(location, "wandb_run_id.txt"), "w") as f:
                f.write(wandb.run.id)

        # If not pretrain, fastforward to rewind step
        if level != "pretrain":
            rewind_time, steps = Time.from_timestring(self.rewinding_steps), 0
            while trainer.state.timer < rewind_time:
                trainer.state.timer.on_batch_complete(train_batch_size)
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

    def _pretrain(self, exp_hash: str, object_store: ObjectStoreProvider) -> None:
        # location
        location = f"{exp_hash}/replicate_{self.replicate}/pretrain/main"

        # if experiment completed, abort
        if utils.object_exists_in_bucket(utils.get_object_name(location, "model_final.pt"), object_store):
            print(f"{utils.get_object_name(location, 'model_final.pt')} exists in bucket")
            return

        # print, make local directory and save hparams
        print("-" * 80 + "\nPretraining\n" + "-" * 80)  # print experiment level
        if os.path.exists(utils.get_local_dir(location)):  # setup local exp dir
            shutil.rmtree(utils.get_local_dir(location))
        os.makedirs(utils.get_local_dir(location))
        with open(utils.get_local_path(location, "hparams.yaml"), "w") as f:  # save hparams
            f.write(self.to_yaml())

        # initialize and save initial model
        seed = self.seed * (self.replicate + 1)  # Adjust seed for replicate
        reproducibility.seed_all(seed)  # Seed rngs before randomly initializing the model
        model = deepcopy(self.model).initialize_object()
        torch.save(model.module.state_dict(), utils.get_local_path(location, "model_init.pt"))

        # set level specific hparams
        train_data = utils.maybe_set_default(self.pretrain_data, default=deepcopy(self.train_data))
        train_batch_size = utils.maybe_set_default(self.pretrain_batch_size, default=self.train_batch_size)
        optimizer = utils.maybe_set_default(self.pretrain_optimizer, default=deepcopy(self.optimizer))
        schedulers = utils.maybe_set_default(self.pretrain_schedulers, default=deepcopy(self.schedulers))
        duration = self.rewinding_steps
        level = "pretrain"

        # train
        self._train(
            exp_hash, level, model, train_data, train_batch_size, optimizer, schedulers, duration, seed, object_store
        )

    def _establish_initial_weights(self, exp_hash: str, object_store: ObjectStoreProvider) -> None:
        # location
        location = f"{exp_hash}/replicate_{self.replicate}/level_0/main"

        # abort if model initialized at level 0
        if utils.object_exists_in_bucket(utils.get_object_name(location, "model_init.pt"), object_store):
            print(f"{utils.get_object_name(location, 'model_init.pt')} exists in bucket")
            return

        # print and make local directory
        print("-" * 80 + "\nEstablishing Initial Weights\n" + "-" * 80)
        if os.path.exists(utils.get_local_dir(location)):  # setup local exp dir
            shutil.rmtree(utils.get_local_dir(location))
        os.makedirs(utils.get_local_dir(location))

        # if there is no pretrain model, initialize and save new model with correct seed
        save_path = utils.get_local_path(location, "model_init.pt")
        if self.rewinding_steps == "0ba":
            seed = self.seed * (self.replicate + 1)  # Adjust seed for replicate
            reproducibility.seed_all(seed)  # Seed rngs before randomly initializing the model
            model = deepcopy(self.model).initialize_object()
            torch.save(model.module.state_dict(), save_path)
        # if there is a pretrain model_final, save it as level_0 model_init
        else:
            pretrain_location = f"{exp_hash}/replicate_{self.replicate}/pretrain/main"
            object_store.download_object(utils.get_object_name(pretrain_location, "model_final.pt"), save_path)

        # upload initial model
        object_store.upload_object(save_path, utils.get_object_name(location, "model_init.pt"))
        shutil.rmtree(utils.get_local_dir(location))

    def _prune_level(self, level: int, exp_hash: str, object_store: ObjectStoreProvider) -> None:
        location = f"{exp_hash}/replicate_{self.replicate}/level_{level}/main"
        if Mask.exists(location, object_store):
            return

        if level == 0:
            Mask.ones_like(deepcopy(self.model).initialize_object()).save(location, object_store)
        else:
            old_location = f"{exp_hash}/replicate_{self.replicate}/level_{level-1}/main"
            state_dict = utils.load_object(old_location, "model_final.pt", object_store, torch.load)
            model = deepcopy(self.model).initialize_object()
            model.module.load_state_dict(state_dict)  # this is now the final model from level-1
            mask = self.pruning.prune(model, Mask.load(old_location, object_store))
            mask.save(location, object_store)

    def _train_level(self, level: int, exp_hash: str, object_store: ObjectStoreProvider) -> None:
        # location
        location = f"{exp_hash}/replicate_{self.replicate}/level_{level}/main"

        # if experiment completed, abort
        if utils.object_exists_in_bucket(utils.get_object_name(location, "model_final.pt"), object_store):
            print(f"{utils.get_object_name(location, 'model_final.pt')} exists in bucket")
            return

        # print and make local directory, save hparams
        print("-" * 80 + f"\nTraining Level {level}\n" + "-" * 80)
        if os.path.exists(utils.get_local_dir(location)):  # setup local exp dir
            shutil.rmtree(utils.get_local_dir(location))
        os.makedirs(utils.get_local_dir(location))
        with open(utils.get_local_path(location, "hparams.yaml"), "w") as f:  # save hparams
            f.write(self.to_yaml())

        # initialize and save initial model
        seed = self.seed * (self.replicate + 1)
        init_location = f"{exp_hash}/replicate_{self.replicate}/level_0/main"
        state_dict = utils.load_object(init_location, "model_init.pt", object_store, torch.load)
        model = deepcopy(self.model).initialize_object()
        model.module.load_state_dict(state_dict)
        pruned_model = PrunedClassifier(model, Mask.load(location, object_store))
        torch.save(pruned_model.module.state_dict(), utils.get_local_path(location, "model_init.pt"))

        # set level specific hparams
        train_data = deepcopy(self.train_data)
        train_batch_size = self.train_batch_size
        optimizer = deepcopy(self.optimizer)
        schedulers = deepcopy(self.schedulers)
        duration = self.max_duration
        level_ = f"level_{level}"

        # train
        self._train(
            exp_hash,
            level_,
            pruned_model,
            train_data,
            train_batch_size,
            optimizer,
            schedulers,
            duration,
            seed,
            object_store,
        )
