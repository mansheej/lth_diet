from __future__ import annotations
from composer.utils import dist
import json
import numpy as np
from numpy.typing import NDArray
import os
import torch


class Mask(dict):
    def __init__(self, other_dict=None) -> None:
        # make a dictionary and merge in other_dict
        super().__init__()
        if other_dict is not None:
            for k, v in other_dict.items():
                self[k] = v

    def __setitem__(self, key: str, value: NDArray | torch.Tensor) -> None:
        # check valid before adding to dict
        if not isinstance(key, str) or len(key) == 0:
            raise ValueError(f"Invalid tensor name: {key}")
        if isinstance(value, np.ndarray):
            value = torch.as_tensor(value)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Value for key {key} must be torch Tensor or numpy ndarray.")
        if ((value != 0) & (value != 1)).any():
            raise ValueError("All entries must be 0 or 1.")
        super().__setitem__(key, value)

    @staticmethod
    def ones_like(model: torch.nn.Module) -> Mask:
        mask = Mask()
        for name in prunable_layer_names(model):  # TODO: abstraction for prunable layer names
            mask[name] = torch.ones(list(model.state_dict()[name].shape))
        return mask

    def save(self, output_location):
        if dist.get_global_rank() != 0:  # only save from rank 0 process
            return
        if not os.path.exists(output_location):
            os.makedirs(output_location)
        save_dict = {k: v.cpu().int() for k, v in self.items()}
        save_path = os.path.join(output_location, "mask.pt")
        torch.save(save_dict, save_path)

        # Create a sparsity report.
        total_weights = np.sum([v.size for v in self.numpy().values()]).item()
        total_unpruned = np.sum([np.sum(v) for v in self.numpy().values()]).item()
        sparsity_report_path = os.path.join(output_location, "sparsity_report.json")
        with open(sparsity_report_path, "w") as fp:
            fp.write(json.dumps({"total": float(total_weights), "unpruned": float(total_unpruned)}, indent=4))

    @staticmethod
    def load(output_location):
        if not Mask.exists(output_location):
            raise ValueError(f"Mask not found at {output_location}")
        mask_path = os.path.join(output_location, "mask.pt")
        return Mask(torch.load(mask_path))

    @staticmethod
    def exists(output_location):
        mask_path = os.path.join(output_location, "mask.pt")
        return os.path.exists(mask_path)

    def numpy(self):
        return {k: v.cpu().numpy() for k, v in self.items()}

    @property
    def sparsity(self):
        # % weights pruned
        unpruned = torch.sum(torch.tensor([torch.sum(v) for v in self.values()]))
        total = torch.sum(torch.tensor([torch.sum(torch.ones_like(v)) for v in self.values()]))
        return 1 - unpruned.float() / total.float()

    @property
    def density(self):
        return 1 - self.sparsity
