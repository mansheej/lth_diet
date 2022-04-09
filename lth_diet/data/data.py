from __future__ import annotations
import abc
from composer.core.types import DataLoader, DataSpec, Dataset
from composer.datasets.dataloader import DataLoaderHparams
from composer.utils import dist
import dataclasses
from lth_diet.data.data_diet import RandomSubset, SubsetByScore
from lth_diet.data.dataset_transform import DatasetTransform
import os
from torch.utils.data import Sampler
from typing import List, Optional
import yahp as hp


dataset_transform_registry = {"random_subset": RandomSubset, "subset_by_score": SubsetByScore}


@dataclasses.dataclass
class DataHparams(hp.Hparams, abc.ABC):
    # Dataset specific params as fields
    hparams_registry = {"dataset_transforms": dataset_transform_registry}
    train: bool = hp.required("Load training data")
    shuffle: Optional[bool] = hp.optional("Default: True if train", default=None)
    drop_last: Optional[bool] = hp.optional("Default: True if train", default=None)
    no_augment: Optional[bool] = hp.optional("Applies to train data, Default: False", default=None)
    dataset_transforms: Optional[List[DatasetTransform]] = hp.optional("List of dataset transforms", default=None)
    data_dir: Optional[str] = hp.optional("Default: use environment variable DATA_DIR", default=None)

    @abc.abstractmethod
    def get_dataset(self, data_dir: str, shuffle: bool, no_augment: bool) -> Dataset:
        ...

    @abc.abstractmethod
    def get_data(
        self, dataset: Dataset, batch_size: int, sampler: Sampler, dataloader_hparams: DataLoaderHparams
    ) -> DataLoader | DataSpec:
        ...

    def initialize_object(
        self, batch_size: int, dataloader_hparams: DataLoaderHparams, **kwargs
    ) -> DataLoader | DataSpec:
        # Non dataset specific params as args, kwargs for optional dataset transforms
        # Set defaults for None fields
        shuffle = self.train if self.shuffle is None else self.shuffle
        drop_last = self.train if self.drop_last is None else self.drop_last
        no_augment = False if self.no_augment is None else self.no_augment
        dataset_transforms = [] if self.dataset_transforms is None else self.dataset_transforms
        data_dir = os.environ["DATA_DIR"] if self.data_dir is None else self.data_dir
        # Get and transform dataset
        dataset = self.get_dataset(data_dir, shuffle, no_augment)
        for transform in dataset_transforms:
            dataset = transform.apply(dataset, **kwargs)
        # Sampler and Dataloader | DataSpec
        sampler = dist.get_sampler(dataset, drop_last=drop_last, shuffle=shuffle)
        data = self.get_data(dataset, batch_size, sampler, dataloader_hparams)
        return data
