from dataclasses import dataclass

import yahp as hp
from composer.models import ModelHparams

from lth_diet.models import model_registry


@dataclass
class TrainExp(hp.Hparams):
    hparams_registry = {"model": model_registry}

    replicate: int = hp.required(doc="Experiment replicate number")
    model_seed: int = hp.required(
        doc="Multiply by replicate+1 to get seed for initializing the model"
    )
    sgd_seed: int = hp.required(doc="Multiply by replicate+1 to get seed for SGD noise")

    model: ModelHparams = hp.required(doc="Model hparams")

    def validate(self) -> None:
        super().validate()
        if self.replicate < 0:
            raise ValueError(f"replicate must be non-negative")
        if self.model_seed <= 0:
            raise ValueError(f"model_seed must by positive")
        if self.sgd_seed <= 0:
            raise ValueError(f"sgd_seed must be positive")

    def run(self) -> None:
        self.validate()
        print(self)
