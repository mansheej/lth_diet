from lth_diet.models.classifier import ClassifierHparams
from lth_diet.models.resnet import ResNetClassifierHparams
from lth_diet.models.resnet_cifar import ResNetCIFARClassifierHparams
from lth_diet.models.vgg_cifar import VGG16CIFARClassifierHparams

model_registry = {"resnet_cifar": ResNetCIFARClassifierHparams, "resnet": ResNetClassifierHparams, "vgg16": VGG16CIFARClassifierHparams}
