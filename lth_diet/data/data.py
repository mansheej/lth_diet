from __future__ import annotations
import abc
from dataclasses import dataclass
from typing import Optional

import yahp as hp
from composer.core.types import DataLoader, DataSpec, Dataset
from composer.datasets.dataloader import DataLoaderHparams
from composer.utils import dist
from torch.utils.data import Sampler


@dataclass
class DataHparams(hp.Hparams, abc.ABC):
    train: bool = hp.required("True: load training set, False: load test set")
    shuffle: bool = hp.required("True: reshuffle dataset every epoch")
    drop_last: bool = hp.required("Last incomplete batch: True: drop, False: pad with zeros")
    datadir: Optional[str] = hp.optional("Path to data directory", default=None)

    @abc.abstractmethod
    def get_dataset(self) -> Dataset:
        pass

    @abc.abstractmethod
    def get_data(
        self, dataset: Dataset, sampler: Sampler, batch_size: int, dataloader_hparams: DataLoaderHparams
    ) -> DataLoader | DataSpec:
        pass

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> DataLoader | DataSpec:
        dataset = self.get_dataset()
        sampler = dist.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)
        data = self.get_data(dataset, sampler=sampler, batch_size=batch_size, dataloader_hparams=dataloader_hparams)
        return data
