from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path

from composer.core.types import DataLoader, DataSpec, Dataset
from composer.datasets.dataloader import DataloaderHparams
from torch.utils.data import Sampler
from torchvision import transforms
from torchvision.datasets import CIFAR10

from lth_diet.data.data import DataHparams
from lth_diet.utils import utils


@dataclass
class CIFAR10DataHparams(DataHparams):
    @property
    def name(self) -> str:
        ignore = ["datadir"]
        return utils.get_hparams_name(self, "cifar10", ignore)

    def get_dataset(self) -> Dataset:
        cifar10_mean, cifar10_std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
        datadir = self.datadir if self.datadir else os.environ["DATADIR"]
        datadir = Path(datadir) / "cifar10"
        if self.train:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
                ]
            )
        dataset = CIFAR10(root=datadir, train=self.train, transform=transform)
        return dataset

    def get_data(
        self,
        dataset: Dataset,
        sampler: Sampler,
        batch_size: int,
        dataloader_hparams: DataloaderHparams,
    ) -> DataLoader | DataSpec:
        data = dataloader_hparams.initialize_object(
            dataset, batch_size=batch_size, sampler=sampler, drop_last=self.drop_last
        )
        return data
