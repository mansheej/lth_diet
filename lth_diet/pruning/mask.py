from composer.utils import dist
import json
import numpy as np
import os
import torch


class Mask(dict):
    def __init__(self, other_dict=None):
        # initialize a dict with items from an optional dict
        super().__init__()
        if other_dict is not None:
            for k, v in other_dict.items():
                self[k] = v

    def __setitem__(self, key, value):
        # key must be a non-empty string
        if not isinstance(key, str) or len(key) == 0:
            raise ValueError(f"Invalid tensor name: {key}")
        # value in: array or tensor, value out: tensor
        if isinstance(value, np.ndarray):
            value = torch.as_tensor(value)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Value for {key} must be a torch Tensor or numpy ndarray")
        # value must be a boolean mask
        if ((value != 0) & (value != 1)).any():
            raise ValueError("All entries must be 0 or 1")
        super().__setitem__(key, value)

    @staticmethod
    def ones_like(model):  # TODO: annotate types
        mask = Mask()
        # tensor of ones for every prunable layer
        for name in model.prunable_layer_names:  # TODO: model.prunable_layer_names
            mask[name] = torch.ones(list(model.state_dict()[name].shape))
        return mask

    def save(self, output_location):
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
    def load(output_location):
        if not Mask.exists(output_location):
            raise ValueError(f"Mask not found at {output_location}")
        return Mask(torch.load(os.path.join(output_location, "mask.pt")))

    @staticmethod
    def exists(output_location):
        return os.path.exists(os.path.join(output_location, "mask.pt"))

    def numpy(self):
        # turn tensors into arrays
        return {k: v.cpu().numpy() for k, v in self.items()}

    @property
    def sparsity(self):
        unpruned = torch.sum(torch.tensor([torch.sum(v) for v in self.values()]))
        total = torch.sum(torch.tensor([torch.sum(torch.ones_like(v)) for v in self.values()]))
        return 1 - unpruned.float() / total.float()

    @property
    def density(self):
        return 1 - self.sparsity
