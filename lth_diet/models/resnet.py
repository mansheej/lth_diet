from composer.models import ComposerClassifier
from dataclasses import dataclass
from lth_diet.models.classifier import ComposerClassifierHparams
from lth_diet.utils import utils
from torch import nn
from torchvision import models
from typing import Optional
import yahp as hp


@dataclass
class ResNetClassifierHparams(ComposerClassifierHparams):
    num_layers: int = hp.required("Number of layers: 18 | 50")
    low_res: Optional[bool] = hp.optional("Low resolution for 32x32x3 images", default=None)

    @property
    def name(self) -> str:
        return utils.get_hparams_name(self, prefix="ResNet")

    def initialize_object(self) -> ComposerClassifier:
        if self.num_layers == 18:
            model = models.resnet18(num_classes=self.num_classes)
        elif self.num_layers == 50:
            model = models.resnet50(num_classes=self.num_classes)
        else:
            raise ValueError("Currently supported: num_layers in [18, 50]")
        if self.low_res:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()
        return ComposerClassifier(module=model)
