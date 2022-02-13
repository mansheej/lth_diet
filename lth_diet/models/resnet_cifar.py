from dataclasses import dataclass
from typing import List

import yahp as hp
from composer.models import ComposerClassifier, Initializer, ModelHparams
from composer.models.resnets import CIFAR_ResNet


@dataclass
class ResNetCIFAR(ModelHparams):
    num_layers: int = hp.optional(doc="Number of layers, supported: 20, 56", default=20)
    num_classes: int = hp.optional(doc="Number of classes", default=10)
    initializers: List[Initializer] = hp.optional(
        doc="The initialization strategy for the model",
        default_factory=lambda: [Initializer.KAIMING_NORMAL, Initializer.BN_UNIFORM],
    )

    def validate(self):
        super().validate()
        if self.num_layers not in [20, 56]:
            raise ValueError("Currently supported: num_layers = 20 | 56")

    def initialize_object(self) -> ComposerClassifier:
        model = CIFAR_ResNet.get_model_from_name(
            model_name=f"cifar_resnet_{self.num_layers}",
            initializers=self.initializers,
            outputs=self.num_classes,
        )
        return ComposerClassifier(module=model)
