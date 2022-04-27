from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
import torch
from composer.core.types import BatchPair
from composer.models import ComposerClassifier
from torch import nn

if TYPE_CHECKING:
    from lth_diet.pruning.mask import Mask


class PrunedClassifier(ComposerClassifier):
    @staticmethod
    def to_mask_name(name: str) -> str:
        return "mask_" + name.replace(".", "___")

    def __init__(self, model: ComposerClassifier, mask: Mask) -> None:
        if isinstance(model, PrunedClassifier):
            raise ValueError("Cannot nest PrunedClassifiers")
        super().__init__(module=model.module)  # PrunedClassifier.module = CompserClassifier.module

        for k in prunable_layer_names(self):  # check prunable parameters have corresponding masks with correct shapes
            if k not in mask:
                raise ValueError(f"Missing mask value {k}")
            if not np.array_equal(mask[k].shape, np.array(self.module.state_dict()[k].shape)):
                raise ValueError(f"Incorrect mask shape {mask[k].shape} for tensor {k}")

        for k in mask:  # check all masks correspond to prunable layers
            if k not in prunable_layer_names(self):
                raise ValueError(f"Key {k} found in mask but is not a valid model tensor")

        for k, v in mask.items():  # register mask as buffers
            self.register_buffer(PrunedClassifier.to_mask_name(k), v.float())
        self._apply_mask()  # mask initialized model

    def _apply_mask(self) -> None:
        # apply masks in buffer to parameters
        for name, param in self.named_parameters():
            if hasattr(self, PrunedClassifier.to_mask_name(name)):
                param.data *= getattr(self, PrunedClassifier.to_mask_name(name))

    def forward(self, batch: BatchPair) -> torch.Tensor:
        self._apply_mask()
        return super().forward(batch)


def prunable_layer_names(model: ComposerClassifier) -> List[str]:
    # names of prunable parameters (weights of Conv2d and Linear modules) of a ComposerClassifier
    names = [
        name + ".weight"
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
    ]
    return names
