# from composer.models import ComposerClassifier
# from lth_diet.pruning import mask

# import numpy as np
# from torch import nn


# class PrunedClassifier(ComposerClassifier):
#     @staticmethod
#     def to_mask_name(name):
#         return "mask_" + name.replace(".", "___")


#     def __init__(self, model: ComposerClassifier, mask: Mask):
#         if isinstance(model, PrunedClassifier):
#             raise ValueError("Cannot nest PrunedClassifiers")

#         # PrunedClassifier.module = CompserClassifier.module
#         super().__init__(module=model.module)

#         # check prunable layers have masks with correct shapes
#         for k in self.prunable_layer_names:
#             if k not in mask:
#                 raise ValueError(f"Missing mask value {k}")
#             if not np.array_equal(mask[k].shape, np.array(self.module.state_dict()[k].shape)):
#                 raise ValueError(f"Incorrect mask shape {mask[k].shape} for tensor {k}")

#         # check masks correspond to prunable layers
#         for k in mask:
#             if k not in self.prunable_layer_names:
#                 raise ValueError(f"Key {k} found in mask but is not a valid model tensor")

#         # register masks as buffers (no gradients) and apply
#         for k, v in mask.items():
#             self.register_buffer(PrunedClassifier.to_mask_name(k), v.float())
#         self._apply_mask()

#     def _apply_mask(self):
#         for name, param in self.module.named_parameters():
#             if hasattr(self, PrunedClassifier.to_mask_name(name)):
#                 param.data *= getattr(self, PrunedClassifier.to_mask_name(name))

#     def forward(self, x):
#         self._apply_mask()
#         return super().forward(x)

#     @property
#     def prunable_layer_names(self):
#         return [
#             name + ".weight"
#             for name, module in self.module.named_modules()
#             if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
#         ]
