from composer.core.types import DataLoader, Dataset
from composer.datasets.dataloader import DataLoaderHparams
from dataclasses import dataclass
from lth_diet.data.data import DataHparams
from lth_diet.utils import utils
import os
from pathlib import Path
from torch.utils.data import Sampler
from torchvision import transforms
from torchvision.datasets import CIFAR100


@dataclass
class CIFAR100DataHparams(DataHparams):
    @property
    def name(self) -> str:
        ignore_fields = ["datadir"]
        return utils.get_hparams_name(self, prefix="CIFAR100", ignore_fields=ignore_fields)

    def get_dataset(self) -> Dataset:
        cifar100_mean, cifar100_std = [0.5071, 0.4867, 0.4408], [0.2673, 0.2564, 0.2762]
        datadir = Path(os.environ["DATADIR"] if self.datadir is None else self.datadir) / "cifar100"
        if self.train and not self.no_augment:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
                ]
            )
        dataset = CIFAR100(root=datadir, train=self.train, transform=transform)
        return dataset

    def get_data(
        self, dataset: Dataset, sampler: Sampler, batch_size: int, dataloader_hparams: DataLoaderHparams
    ) -> DataLoader:
        data = dataloader_hparams.initialize_object(
            dataset, batch_size=batch_size, sampler=sampler, drop_last=self.drop_last
        )
        return data
