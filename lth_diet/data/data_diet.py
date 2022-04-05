from composer.core.types import Dataset
from composer.utils.object_store import ObjectStoreProvider
from composer.utils.run_directory import get_run_directory
from coolname import generate_slug
from dataclasses import dataclass
from lth_diet.data.dataset_transform import DatasetTransform
from lth_diet.utils import utils
import numpy as np
from numpy.typing import NDArray
import os
from torch.utils.data import Subset
from typing import Optional
import yahp as hp


@dataclass
class RandomSubset(DatasetTransform):
    size: Optional[int] = hp.optional("Number of examples in subset, None: all examples", default=None)
    class_balanced: Optional[bool] = hp.optional("True: class balanced sampling", default=None)
    seed: Optional[int] = hp.optional("Seed for random subset, None: 42", default=None)
    replicate_seed: Optional[bool] = hp.optional(
        "True: use same seed for all replicates, None | False: seed = seed * (replicate + 1)", default=None
    )

    def _get_class_balanced_subset(self, dataset: Dataset, size: int, seed: int) -> Dataset:
        num_classes = len(dataset.classes)
        size_per_class = size // num_classes
        size_rand = size - (size_per_class * num_classes)
        idxs, idxs_left = [], []
        for c in range(num_classes):
            c_idxs = np.arange(len(dataset))[np.array(dataset.targets) == c]
            c_idxs = np.random.RandomState(seed + c).permutation(c_idxs)
            idxs.append(c_idxs[:size_per_class])
            idxs_left.append(c_idxs[size_per_class:])
        idxs_left = np.concatenate(idxs_left)
        idxs.append(np.random.RandomState(seed + num_classes).choice(idxs_left, size_rand, replace=False))
        idxs = np.sort(np.concatenate(idxs))
        dataset = Subset(dataset, idxs)
        return dataset

    def _get_subset(self, dataset: Dataset, size: int, seed: int) -> Dataset:
        idxs = np.sort(np.random.RandomState(seed).choice(len(dataset), size, replace=False))
        dataset = Subset(dataset, idxs)
        return dataset

    def apply(self, dataset: Dataset, replicate: int = None, **kwargs) -> Dataset:
        size = len(dataset) if self.size is None else self.size
        seed = 42 if self.seed is None else self.seed  # set seed to 42 if not provided
        if not self.replicate_seed:  # if replicate_seed=True, seed is the same for all replicates
            seed = seed * (replicate + 1)  # if replicate_seed=False|None, set different seed per replicate
        if self.class_balanced:
            dataset = self._get_class_balanced_subset(dataset, size, seed)
        else:
            dataset = self._get_subset(dataset, size, seed)
        return dataset


@dataclass
class SubsetByScore(DatasetTransform):
    score: str = hp.required("Scores (in bucket) for sorting examples in ascending order")
    size: Optional[int] = hp.optional("Number of examples in subset, None: all examples", default=None)
    left_offset: Optional[int] = hp.optional("Number of examples to offset from left, None: 0", default=None)
    right_offset: Optional[int] = hp.optional("Number of examples to offset from right", default=None)
    class_balanced: Optional[bool] = hp.optional(
        "True: class balanced sampling (offset must be divisible by number of classes)", default=None
    )

    def _get_class_balanced_subset(
        self, dataset: Dataset, size: int, offset: int, sort_idxs: NDArray[np.int64]
    ) -> Dataset:
        num_classes = len(dataset.classes)
        size_per_class, offset_per_class = size // num_classes, offset // num_classes
        size_left, offset_left = size - size_per_class * num_classes, offset - offset_per_class * num_classes
        assert offset_left == 0, "If class balanced, offset must be divisible by number of classes"
        sorted_targets = np.array(dataset.targets)[sort_idxs]  # sort targets by score (sort index)
        idxs = []
        for c in range(num_classes):
            c_idxs = sort_idxs[sorted_targets == c]  # indices for class c sorted by score
            end = offset_per_class + size_per_class
            if c < size_left:
                end += 1  # first size_left classes get an extra example to ensure examples sum to total size
            idxs.append(c_idxs[offset_per_class:end])
        idxs = np.sort(np.concatenate(idxs))
        dataset = Subset(dataset, idxs)
        return dataset

    def _get_subset(self, dataset: Dataset, size: int, offset: int, sort_idxs: NDArray[np.int64]) -> Dataset:
        idxs = np.sort(sort_idxs[offset : offset + size])
        dataset = Subset(dataset, idxs)
        return dataset

    def apply(self, dataset: Dataset, object_store: ObjectStoreProvider, **kwargs) -> Dataset:
        object_name = f"exps/scores/{self.score}"
        assert utils.object_exists_in_bucket(object_name, object_store), "Cannot find score in bucket"
        assert not (
            self.left_offset is not None and self.right_offset is not None
        ), "Left and right offset are mutually exclusive, cannot specify both"
        os.makedirs(os.path.join(get_run_directory(), "scores"), exist_ok=True)
        score_path = os.path.join(get_run_directory(), f"scores/{generate_slug()}_{self.score}")
        object_store.download_object(object_name, score_path)
        scores = np.load(score_path)
        sort_idxs = np.argsort(scores)
        size = len(dataset) if self.size is None else self.size
        if self.right_offset is None:
            offset = 0 if self.left_offset is None else self.left_offset
        else:
            offset = self.right_offset
            sort_idxs = sort_idxs[::-1]
        if self.class_balanced:
            dataset = self._get_class_balanced_subset(dataset, size, offset, sort_idxs)
        else:
            dataset = self._get_subset(dataset, size, offset, sort_idxs)
        os.remove(score_path)
        return dataset
