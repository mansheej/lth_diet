from composer.core.types import Dataset, DataLoader
from composer.datasets.dataloader import DataLoaderHparams
import dataclasses
from lth_diet.data.data import DataHparams
from lth_diet.utils import utils
import os
from torch.utils.data import Sampler
import torchvision
from torchvision import transforms


CINIC10_MEAN, CINIC10_STD = [0.4789, 0.4723, 0.4305], [0.2421, 0.2383, 0.2587]


@dataclasses.dataclass
class CINIC10DataHparams(DataHparams):
    @property
    def name(self) -> str:
        return utils.get_hparams_name(self, prefix="CINIC10", ignore_fields=["data_dir"])

    def get_dataset(self, data_dir: str, no_augment: bool) -> Dataset:
        data_dir = os.path.join(data_dir, "cinic10")
        if self.train and not no_augment:  # augment if training data and no_augment is False
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(CINIC10_MEAN, CINIC10_STD),
                ]
            )
        else:  # no augmentation if test data or no_augment is True
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(CINIC10_MEAN, CINIC10_STD),
                ]
            )
        # Only checks if self.train is True
        if self.train:
            dataset = torchvision.datasets.ImageFolder(data_dir + '/train', transform=transform)
        else:
            dataset = torchvision.datasets.ImageFolder(data_dir + '/test', transform=transform)
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
