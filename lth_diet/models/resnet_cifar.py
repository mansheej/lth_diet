from composer.models import ComposerClassifier, Initializer
from composer.models.resnets import CIFAR_ResNet
import dataclasses
from lth_diet.models.classifier import ClassifierHparams
from lth_diet.utils import utils
import yahp as hp


@dataclasses.dataclass
class ResNetCIFARClassifierHparams(ClassifierHparams):
    num_layers: int = hp.required("Number of layers: 20 | 56")

    @property
    def name(self) -> str:
        return utils.get_hparams_name(self, prefix="ResNetCIFAR")

    def initialize_object(self) -> ComposerClassifier:
        model = CIFAR_ResNet.get_model_from_name(
            model_name=f"cifar_resnet_{self.num_layers}",
            initializers=[Initializer.KAIMING_NORMAL, Initializer.BN_UNIFORM],
            outputs=self.num_classes,
        )
        return ComposerClassifier(model)
