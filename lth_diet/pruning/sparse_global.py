import dataclasses
from typing import Optional

import numpy as np
import yahp as hp
from composer.models import ComposerClassifier
from lth_diet.pruning import Mask
from lth_diet.pruning.pruned_classifier import prunable_layer_names


@dataclasses.dataclass
class PruningHparams(hp.Hparams):
    pruning_fraction: float = hp.optional("Fraction of additional weights to prune", default=0.2)
    pruning_layers_to_ignore: Optional[str] = hp.optional("Comma-separated list of tensors not pruned", default=None)

    def prune(self, model: ComposerClassifier, current_mask: Optional[Mask] = None) -> Mask:
        # if no current_mask, make a mask of ones and convert everything to numpy
        current_mask = Mask.ones_like(model).numpy() if current_mask is None else current_mask.numpy()

        # determine number of weights that need to be
        number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask.values()])
        number_of_weights_to_prune = np.ceil(self.pruning_fraction * number_of_remaining_weights).astype(int)

        # determine which layers can be pruned.
        prunable_tensors = set(prunable_layer_names(model))
        if self.pruning_layers_to_ignore:
            prunable_tensors -= set(self.pruning_layers_to_ignore.split(","))

        # Get the model weights.
        weights = {k: v.clone().cpu().detach().numpy() for k, v in model.state_dict().items() if k in prunable_tensors}

        # Vector of all unpruned weights are sorted and used to determine a threshold
        weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in weights.items()])
        threshold = np.sort(np.abs(weight_vector))[number_of_weights_to_prune]

        # New mask by thresholding weights
        new_mask = Mask(
            {k: np.where(np.abs(v) > threshold, current_mask[k], np.zeros_like(v)) for k, v in weights.items()}
        )

        # Fill in any items of current_mask that got dropped
        for k in current_mask:
            if k not in new_mask:
                new_mask[k] = current_mask[k]

        return new_mask
