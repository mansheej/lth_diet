from lth_diet.data.data import DataHparams
from lth_diet.data.cifar10 import CIFAR10DataHparams
from lth_diet.data.cifar100 import CIFAR100DataHparams

data_registry = {"cifar10": CIFAR10DataHparams, "cifar100": CIFAR100DataHparams}
