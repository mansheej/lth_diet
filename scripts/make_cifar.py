from pathlib import Path
from torchvision.datasets import CIFAR10, CIFAR100

ROOT = Path("/home/mansheej/lth_diet")  # path to repo
CIFAR10(ROOT / "data/cifar10", download=True)
CIFAR100(ROOT / "data/cifar100", download=True)
