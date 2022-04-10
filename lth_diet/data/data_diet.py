from composer.core.types import Dataset
from composer.utils.object_store import ObjectStoreProvider
import dataclasses
import io
from lth_diet.data.dataset_transform import DatasetTransform
from lth_diet.utils import utils
import numpy as np
from numpy.typing import NDArray
import os
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
    score: str = hp.required("Scores used for sorting in ascending order (in object store)")
    size: Optional[int] = hp.optional("Default: all examples", default=None)
    left_offset: Optional[int] = hp.optional("Default: 0", default=None)
    right_offset: Optional[int] = hp.optional("Mutually exclusive with left_offset", default=None)
    class_balanced: Optional[bool] = hp.optional("Default: False", default=None)

    def _get_class_balanced_subset(
        self, dataset: Dataset, size: int, offset: int, sort_idxs: NDArray[np.int64]
    ) -> Dataset:
        # Class balancing stats
        num_classes = len(dataset.classes)
        size_per_class, size_left = size // num_classes, size % num_classes
        offset_per_class, offset_left = offset // num_classes, offset % num_classes
        assert offset_left == 0, "If class balanced, offset must be divisible by number of classes"
        # Sort targets by score to match sort_idxs
        sorted_targets = np.array(dataset.targets)[sort_idxs]
        # Sample indices from each class
        idxs = []
        for c in range(num_classes):
            c_idxs = sort_idxs[sorted_targets == c]  # Indices of examples of a class sorted by score
            end = offset_per_class + size_per_class
            if c < size_left:  # Ensure that subset has the right total size after sampling evenly across classes
                end += 1  # Each class gets an extra example until we have chosen the remaining size_left examples
            idxs.append(c_idxs[offset_per_class:end])
        idxs = np.sort(np.concatenate(idxs))  # sort examples in original order
        dataset = Subset(dataset, idxs)
        return dataset

    def _get_subset(self, dataset: Dataset, size: int, offset: int, sort_idxs: NDArray[np.int64]) -> Dataset:
        idxs = np.sort(sort_idxs[offset : offset + size])  # sort examples in original order
        dataset = Subset(dataset, idxs)
        return dataset

    def apply(self, dataset: Dataset, object_store: ObjectStoreProvider, **kwargs) -> Dataset:
        # Validate offsets
        assert not (
            self.left_offset is not None and self.right_offset is not None
        ), "left_offset and right_offset are mutually exclusive, cannot specify both"
        # Validate score
        score_object_name = f"{os.environ['OBJECT_STORE_DIR']}/scores/{self.score}.npy"
        assert utils.object_exists_in_bucket(score_object_name, object_store), "Cannot find score in bucket"
        # Set defaults for None fields
        size = utils.maybe_set_default(self.size, default=len(dataset))
        class_balanced = utils.maybe_set_default(self.class_balanced, default=False)
        # download and sort scores
        scores = np.load(io.BytesIO(next(object_store.download_object_as_stream(score_object_name))))
        sort_idxs = np.argsort(scores)  # example indices in ascending order by score
        # set offset and order
        if self.right_offset is None:  # left offset may or may not be None
            offset = utils.maybe_set_default(self.left_offset, default=0)
        else:  # right_offset is not None and left_offset is None
            offset = self.right_offset
            sort_idxs = sort_idxs[::-1]  # flip scores so that the subset span is always [offset, offset+size]
        # subsample
        if class_balanced:
            dataset = self._get_class_balanced_subset(dataset, size, offset, sort_idxs)
        else:
            dataset = self._get_subset(dataset, size, offset, sort_idxs)
        return dataset
