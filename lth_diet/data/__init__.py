from lth_diet.data.data import DataHparams
from lth_diet.data.cifar import CIFAR10DataHparams, CIFAR100DataHparams

data_registry = {"cifar10": CIFAR10DataHparams, "cifar100": CIFAR100DataHparams}
