from lth_diet.models.classifier import ClassifierHparams
from lth_diet.models.resnet import ResNetClassifierHparams
from lth_diet.models.resnet_cifar import ResNetCIFARClassifierHparams
from lth_diet.models.vgg import VGG16ClassifierHparams

model_registry = {
    "resnet_cifar": ResNetCIFARClassifierHparams,
    "resnet": ResNetClassifierHparams,
    "vgg16": VGG16ClassifierHparams,
}
