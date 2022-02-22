from dataclasses import dataclass
from typing import List

import yahp as hp
from composer.models import ComposerClassifier, Initializer
from composer.models.resnets import CIFAR_ResNet

from lth_diet.models.classifier import ClassifierHparams
from lth_diet.utils import utils


@dataclass
class ResNetCIFARClassifierHparams(ClassifierHparams):
    num_layers: int = hp.required("Number of layers: 20 | 56")
    initializers: List[Initializer] = hp.optional(
        "Default: kaiming_normal, bn_uniform",
        default_factory=lambda: [Initializer.KAIMING_NORMAL, Initializer.BN_UNIFORM],
    )

    @property
    def name(self) -> str:
        return utils.get_hparams_name(self, "ResNetCIFAR", [])

    def validate(self):
        super().validate()
        if self.num_layers not in [20, 56]:
            raise ValueError("Currently supported: num_layers in [20, 56]")

    def initialize_object(self) -> ComposerClassifier:
        model = CIFAR_ResNet.get_model_from_name(
            model_name=f"cifar_resnet_{self.num_layers}",
            initializers=self.initializers,
            outputs=self.num_classes,
        )
        return ComposerClassifier(module=model)
