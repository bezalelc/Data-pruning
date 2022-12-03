"""
helpers for common datasets operations
"""

import os

import torchvision
from torchvision import transforms

PATH_DATASET = os.path.abspath(r'../../../datasets')


def get_cifar10():
    """
    get cifar10 train set and test set

    Returns:
        cifar10 train set and test set
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    ])

    # train mean = [0.49139968, 0.48215841, 0.44653091]
    # train std  = [0.24703223, 0.24348513, 0.26158784]

    dataset_train = torchvision.datasets.CIFAR10(PATH_DATASET, train=True, transform=transform, download=True)
    dataset_test = torchvision.datasets.CIFAR10(PATH_DATASET, train=False, transform=transform, download=True)
    return dataset_train, dataset_test
