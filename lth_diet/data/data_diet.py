from composer.core.types import Dataset
import dataclasses
from lth_diet.data.dataset_transform import DatasetTransform
from lth_diet.utils import utils
import numpy as np
from torch.utils.data import Subset
from typing import Optional
import yahp as hp


@dataclasses.dataclass
class RandomSubset(DatasetTransform):
    size: Optional[int] = hp.optional("Default: all examples", default=None)
    seed: Optional[int] = hp.optional("Default: 42", default=None)
    class_balanced: Optional[bool] = hp.optional("Default: False", default=None)
    replicate_seed: Optional[bool] = hp.optional("All replicates get same seed, Default: False", default=None)

    def _get_class_balanced_subset(self, dataset: Dataset, size: int, seed: int) -> Dataset:
        # class balancing stats
        num_classes = len(dataset.classes)
        size_per_class, size_rand = size // num_classes, size % num_classes
        # split indices into keep and extra
        rng = np.random.RandomState(seed)
        idxs, idxs_extra = [], []
        for c in range(num_classes):
            c_idxs = rng.permutation(np.arange(len(dataset))[np.array(dataset.targets) == c])
            idxs.append(c_idxs[:size_per_class])
            idxs_extra.append(c_idxs[size_per_class:])
        # sample remaining from extra indices
        idxs.append(rng.choice(np.concatenate(idxs_extra), size_rand, replace=False))
        # sort examples in original order
        idxs = np.sort(np.concatenate(idxs))
        dataset = Subset(dataset, idxs)
        return dataset

    def _get_subset(self, dataset: Dataset, size: int, seed: int) -> Dataset:
        # sort examples in original order
        idxs = np.sort(np.random.RandomState(seed).choice(len(dataset), size, replace=False))
        dataset = Subset(dataset, idxs)
        return dataset

    def apply(self, dataset: Dataset, replicate: int, **kwargs) -> Dataset:
        # Set defaults for None fields
        size = utils.maybe_set_default(self.size, default=len(dataset))
        seed = utils.maybe_set_default(self.seed, default=42)
        class_balanced = utils.maybe_set_default(self.class_balanced, default=False)
        replicate_seed = utils.maybe_set_default(self.replicate_seed, default=False)
        # set seed for replicate
        if not replicate_seed:
            seed = seed * (replicate + 1)
        # subsample
        if class_balanced:
            dataset = self._get_class_balanced_subset(dataset, size, seed)
        else:
            dataset = self._get_subset(dataset, size, seed)
        return dataset


@dataclasses.dataclass
class SubsetByScore(DatasetTransform):
    pass
