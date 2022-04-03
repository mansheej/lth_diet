import abc
from composer.core.types import Dataset
from dataclasses import dataclass
import yahp as hp


@dataclass
class DatasetTransform(hp.Hparams, abc.ABC):
    @abc.abstractmethod
    def apply(self, dataset: Dataset, **kwargs) -> Dataset:
        pass
