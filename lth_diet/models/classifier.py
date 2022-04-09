import abc
from composer.models import ComposerClassifier
import dataclasses
import yahp as hp


@dataclasses.dataclass
class ClassifierHparams(hp.Hparams, abc.ABC):
    num_classes: int = hp.required("Number of classes")

    @abc.abstractmethod
    def initialize_object(self) -> ComposerClassifier:
        ...
