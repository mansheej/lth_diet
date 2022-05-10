from lth_diet.data.data import DataHparams
from lth_diet.data.cifar import CIFAR10DataHparams, CIFAR100DataHparams 
from lth_diet.data.cinic import CINIC10DataHparams

data_registry = {"cifar10": CIFAR10DataHparams, "cifar100": CIFAR100DataHparams, "cinic10": CINIC10DataHparams}
