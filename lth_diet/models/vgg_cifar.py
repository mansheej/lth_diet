from composer.models import ComposerClassifier, Initializer
from torchvision.models import vgg16
import dataclasses
from lth_diet.models.classifier import ClassifierHparams
from lth_diet.utils import utils
import yahp as hp


@dataclasses.dataclass
class VGG16CIFARClassifierHparams(ClassifierHparams):

    @property
    def name(self) -> str:
        return utils.get_hparams_name(self, prefix="VGG16CIFAR")

    def initialize_object(self) -> ComposerClassifier:
        model = vgg16(
            num_classes=self.num_classes,
        )
        return ComposerClassifier(model)
