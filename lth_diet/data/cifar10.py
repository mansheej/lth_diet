from composer.core.types import Dataset, DataLoader
from composer.datasets.dataloader import DataLoaderHparams
import dataclasses
from lth_diet.data.data import DataHparams
from lth_diet.utils import utils
import os
from torch.utils.data import Sampler
from torchvision import transforms
from torchvision.datasets import CIFAR10


CIFAR10_MEAN, CIFAR10_STD = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]


@dataclasses.dataclass
class CIFAR10DataHparams(DataHparams):
    @property
    def name(self) -> str:
        return utils.get_hparams_name(self, prefix="CIFAR10", ignore_fields=["data_dir"])

    def get_dataset(self, data_dir: str, no_augment: bool) -> Dataset:
        data_dir = os.path.join(data_dir, "cifar10")
        if self.train and not no_augment:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                ]
            )
        dataset = CIFAR10(root=data_dir, train=self.train, transform=transform)
        return dataset

    def get_data(
        self,
        dataset: Dataset,
        batch_size: int,
        sampler: Sampler,
        drop_last: bool,
        dataloader_hparams: DataLoaderHparams,
    ) -> DataLoader:
        data = dataloader_hparams.initialize_object(
            dataset, batch_size=batch_size, sampler=sampler, drop_last=drop_last
        )
        return data
