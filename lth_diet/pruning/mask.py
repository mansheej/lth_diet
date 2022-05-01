from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
import torch
from composer.models import ComposerClassifier
from composer.utils import ObjectStoreProvider
from lth_diet.pruning.pruned_classifier import prunable_layer_names
from lth_diet.utils import utils
from numpy.typing import NDArray


class Mask(dict):
    def __init__(self, other_dict: Dict = None) -> None:
        # dict initialized optionally with type checked items from another dict
        super().__init__()
        if other_dict is not None:
            for k, v in other_dict.items():
                self[k] = v

    def __setitem__(self, key: str, value: torch.Tensor | NDArray) -> None:
        # type check key and value before adding to dict
        if not isinstance(key, str) or len(key) == 0:  # key must be a non-empty string
            raise ValueError(f"Invalid tensor name: {key}")
        if isinstance(value, np.ndarray):  # input value can be Tensor or NDArray, if NDArray, convert to Tensor
            value = torch.as_tensor(value)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Value for {key} must be a torch Tensor or numpy ndarray")
        if ((value != 0) & (value != 1)).any():  # elements of value must be 0 or 1
            raise ValueError("All entries must be 0 or 1")
        super().__setitem__(key, value)

    @staticmethod
    def ones_like(model: ComposerClassifier) -> Mask:
        # Mask containing tensor of ones of the corresponding shape for each prunable parameter
        mask = Mask()
        for name in prunable_layer_names(model):
            mask[name] = torch.ones(list(model.state_dict()[name].shape))
        return mask

    def save(self, location: str, object_store: ObjectStoreProvider) -> None:
        # save as integer cpu tensors
        utils.save_object({k: v.cpu().int() for k, v in self.items()}, location, "mask.pt", object_store, torch.save)
        # create sparsity report
        total_weights = np.sum([v.size for v in self.numpy().values()]).item()
        total_unpruned = np.sum([np.sum(v) for v in self.numpy().values()]).item()

        def write_file(s: str, f: str):
            with open(f, "w") as fp:
                fp.write(s)

        utils.save_object(
            json.dumps({"total": float(total_weights), "unpruned": float(total_unpruned)}, indent=4),
            location,
            "sparsity_report.json",
            object_store,
            write_file,
        )

    @staticmethod
    def load(location: str, object_store: ObjectStoreProvider) -> Mask:
        mask = Mask(utils.load_object(location, "mask.pt", object_store, torch.load))
        return mask

    @staticmethod
    def exists(location: str, object_store: ObjectStoreProvider) -> bool:
        return utils.object_exists_in_bucket(utils.get_object_name(location, "mask.pt"), object_store)

    def numpy(self) -> Dict[str, NDArray]:
        # turn tensors into NDArrays
        return {k: v.cpu().numpy() for k, v in self.items()}

    @property
    def sparsity(self) -> float:
        # fraction of weights pruned
        unpruned = torch.sum(torch.tensor([torch.sum(v) for v in self.values()]))
        total = torch.sum(torch.tensor([torch.sum(torch.ones_like(v)) for v in self.values()]))
        return 1 - unpruned.float() / total.float()

    @property
    def density(self) -> float:
        # fraction of weights remaining
        return 1 - self.sparsity
