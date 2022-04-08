import abc
from composer.core.types import Dataset
import dataclasses
import yahp as hp


@dataclasses.dataclass
class DatasetTransform(hp.Hparams, abc.ABC):
    # Define optional fields in subclasses
    @abc.abstractmethod
    def apply(self, dataset: Dataset, **kwargs) -> Dataset:
        # kwargs contain params only accessible when applying the transform
        ...
