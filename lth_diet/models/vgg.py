import dataclasses
import torch.nn as nn
import torch.nn.functional as F
from composer.models import ComposerClassifier
from lth_diet.models.classifier import ClassifierHparams
from lth_diet.utils import utils


class ConvModule(nn.Module):
    """A single convolutional module in a VGG network."""

    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class VGG(nn.Module):
    """A VGG-style neural network designed for CIFAR-10."""

    def __init__(self, plan, outputs=10):
        super().__init__()
        layers = []
        filters = 3
        for spec in plan:
            if spec == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(ConvModule(filters, spec))
                filters = spec

        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, outputs)

    def forward(self, x):
        x = self.layers(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@dataclasses.dataclass
class VGG16ClassifierHparams(ClassifierHparams):
    @property
    def name(self) -> str:
        return utils.get_hparams_name(self, prefix="VGG16")

    def initialize_object(self) -> ComposerClassifier:
        model = VGG(
            plan=[64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512],
            outputs=self.num_classes,
        )
        return ComposerClassifier(model)
