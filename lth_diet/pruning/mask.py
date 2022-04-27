from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
import torch
from composer.models import ComposerClassifier
from composer.utils import dist
from lth_diet.pruning.pruned_classifier import prunable_layer_names
from numpy.typing import NDArray


class Mask(dict):
    def __init__(self, other_dict: Dict = None) -> None:
        # Dict initialized optionally with type checked items from another Dict
        super().__init__()
        if other_dict is not None:
            for k, v in other_dict.items():
                self[k] = v

    def __setitem__(self, key: str, value: torch.Tensor | NDArray) -> None:
        # type check key and value
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

    def save(self, output_location: str) -> None:
        # only save from rank 0
        if dist.get_global_rank() != 0:
            return
        # make save directory if it doesn't already exist
        if not os.path.exists(output_location):
            os.makedirs(output_location)
        # save as integer cpu tensors
        torch.save({k: v.cpu().int() for k, v in self.items()}, os.path.join(output_location, "mask.pt"))
        # create sparsity report
        total_weights = np.sum([v.size for v in self.numpy().values()]).item()
        total_unpruned = np.sum([np.sum(v) for v in self.numpy().values()]).item()
        with open(os.path.join(output_location, "sparsity_report.json"), "w") as fp:
            fp.write(json.dumps({"total": float(total_weights), "unpruned": float(total_unpruned)}, indent=4))

    @staticmethod
    def load(output_location: str) -> Mask:
        if not Mask.exists(output_location):
            raise ValueError(f"Mask not found at {output_location}")
        # load dict and initialize as mask
        return Mask(torch.load(os.path.join(output_location, "mask.pt")))

    @staticmethod
    def exists(output_location: str) -> bool:
        return os.path.exists(os.path.join(output_location, "mask.pt"))

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
