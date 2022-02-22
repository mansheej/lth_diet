import abc
from dataclasses import dataclass

import yahp as hp
from composer.models import ComposerClassifier


@dataclass
class ClassifierHparams(hp.Hparams, abc.ABC):
    num_classes: int = hp.required("Number of classes")

    @abc.abstractmethod
    def initialize_object(self) -> ComposerClassifier:
        pass
