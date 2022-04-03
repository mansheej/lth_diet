from __future__ import annotations
import abc
from composer.core.types import DataLoader, DataSpec, Dataset
from composer.datasets.dataloader import DataLoaderHparams
from composer.utils import dist
from dataclasses import dataclass
from lth_diet.data.dataset_transform import DatasetTransform
from lth_diet.data.data_diet import RandomSubset, SubsetByScore
from torch.utils.data import Sampler
from typing import List, Optional
import yahp as hp


dataset_transform_registry = {"random_subset": RandomSubset, "subset_by_score": SubsetByScore}
hparams_registry = {"dataset_transforms": dataset_transform_registry}


@dataclass
class DataHparams(hp.Hparams, abc.ABC):
    hparams_registry = hparams_registry
    train: bool = hp.required("True: load training set, False: load test set")
    shuffle: bool = hp.required("True: reshuffle dataset every epoch")
    drop_last: bool = hp.required("True: drop last incomplete batch, False: pad last incomplete batch with zeros")
    no_augment: Optional[bool] = hp.optional("True: do not augment, False | None: augment train data", default=None)
    dataset_transforms: Optional[List[DatasetTransform]] = hp.optional("List of dataset transformations", default=None)
    datadir: Optional[str] = hp.optional("Path to data directory", default=None)

    @abc.abstractmethod
    def get_dataset(self) -> Dataset:
        pass

    @abc.abstractmethod
    def get_data(
        self, dataset: Dataset, sampler: Sampler, batch_size: int, dataloader_hparams: DataLoaderHparams
    ) -> DataLoader | DataSpec:
        pass

    def initialize_object(
        self, batch_size: int, dataloader_hparams: DataLoaderHparams, **kwargs
    ) -> DataLoader | DataSpec:
        dataset = self.get_dataset()
        dataset_transforms = [] if self.dataset_transforms is None else self.dataset_transforms
        for dst in dataset_transforms:
            dataset = dst.apply(dataset, **kwargs)
        sampler = dist.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)
        data = self.get_data(dataset, sampler=sampler, batch_size=batch_size, dataloader_hparams=dataloader_hparams)
        return data
