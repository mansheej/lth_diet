from __future__ import annotations
import abc
from composer.core.types import DataLoader, DataSpec, Dataset
from composer.datasets.dataloader import DataLoaderHparams
from composer.utils import dist
import dataclasses
from lth_diet.data.data_diet import RandomSubset, SubsetByScore
from lth_diet.data.dataset_transform import DatasetTransform
from lth_diet.utils import utils
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
    def get_dataset(self, data_dir: str, no_augment: bool) -> Dataset:
        ...

    @abc.abstractmethod
    def get_data(
        self,
        dataset: Dataset,
        batch_size: int,
        sampler: Sampler,
        drop_last: bool,
        dataloader_hparams: DataLoaderHparams,
    ) -> DataLoader | DataSpec:
        ...

    def initialize_object(
        self, batch_size: int, dataloader_hparams: DataLoaderHparams, **kwargs
    ) -> DataLoader | DataSpec:
        # Non dataset specific params as args, kwargs for optional dataset transforms
        # Set defaults for None fields
        shuffle = utils.maybe_set_default(self.shuffle, default=self.train)
        drop_last = utils.maybe_set_default(self.drop_last, default=self.train)
        no_augment = utils.maybe_set_default(self.no_augment, default=False)
        dataset_transforms = utils.maybe_set_default(self.dataset_transforms, default=[])
        data_dir = utils.maybe_set_default(self.data_dir, default=os.environ["DATA_DIR"])
        # Get and transform dataset
        dataset = self.get_dataset(data_dir, no_augment)
        for transform in dataset_transforms:
            dataset = transform.apply(dataset, **kwargs)
        # Sampler and Dataloader | DataSpec
        sampler = dist.get_sampler(dataset, drop_last=drop_last, shuffle=shuffle)
        data = self.get_data(dataset, batch_size, sampler, drop_last, dataloader_hparams)
        return data
