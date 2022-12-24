"""
helpers for common datasets operations
"""

import os
from typing import Union

import torch
import torchvision
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from ..config import PATH_DATASETS

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


def get_cifar(path_dataset: str, cifar100: bool = True) -> \
        tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
    """
    get cifar10 train set and test set

    Returns:
        cifar10 train set and test set
    """
    if cifar100:
        transform_train = transforms.Compose([
            # transforms.RandomAdjustSharpness(sharpness_factor=2),
            # transforms.ColorJitter(brightness=.6, hue=.04),  # +
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        ])
        dataset_train = torchvision.datasets.CIFAR100(PATH_DATASETS, train=True, transform=transform_train,
                                                      download=True)
        dataset_test = torchvision.datasets.CIFAR100(PATH_DATASETS, train=False, transform=transform_test,
                                                     download=True)

        dataset_train_ordered = torchvision.datasets.CIFAR100(path_dataset, train=True, download=False,
                                                              transform=transform_test)
        dataset_train_raw = torchvision.datasets.CIFAR100(path_dataset, train=True, download=False)
    else:
        transform = transforms.Compose([
            # transforms.Pad(4),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
        ])
        # train mean = [0.49139968, 0.48215841, 0.44653091]
        # train std  = [0.24703223, 0.24348513, 0.26158784]
        dataset_train = torchvision.datasets.CIFAR10(path_dataset, train=True, transform=transform, download=True)
        dataset_test = torchvision.datasets.CIFAR10(path_dataset, train=False, transform=transform, download=True)

        dataset_train_ordered = torchvision.datasets.CIFAR10(path_dataset, train=True, download=False,
                                                              transform=transform)
        dataset_train_raw = torchvision.datasets.CIFAR10(path_dataset, train=True, download=False)

    return dataset_train, dataset_test, dataset_train_ordered, dataset_train_raw


class GPUDataset(Dataset):
    # PATH_DATASETS: str = os.path.abspath(os.path.join('../../../', 'datasets'))
    PATH_TRANSFORMED: str = os.path.abspath(os.path.join(PATH_DATASETS, 'gpu', 'cifar{}_transformed.pt'))
    # NORM_100 = ((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047))
    NORM_10 = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    NORM_05 = ((.5, .5, .5), (.5, .5, .5))

    def __init__(self, load: bool = False, cifar100: bool = True) -> None:
        super().__init__()

        if load:
            dataset = torch.load(
                self.PATH_TRANSFORMED.format('100' if cifar100 else '10').format('100' if cifar100 else '10'))
            self.data, self.targets, self.classes = dataset['x'], dataset['y'], dataset['classes']
        else:
            if cifar100:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
                ])
                dataset_train = torchvision.datasets.CIFAR100(PATH_DATASETS, train=True, transform=transform,
                                                              download=True)
                dataset_test = torchvision.datasets.CIFAR100(PATH_DATASETS, train=False, transform=transform,
                                                             download=True)
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(*self.NORM_10)
                ])
                dataset_train = torchvision.datasets.CIFAR10(PATH_DATASETS, train=True, transform=transform,
                                                             download=True)
                dataset_test = torchvision.datasets.CIFAR10(PATH_DATASETS, train=False, transform=transform,
                                                            download=True)
            self.data: Tensor = torch.cat(
                (torch.stack([dataset_train[i][0] for i in range(len(dataset_train))], dim=0),
                 torch.stack([dataset_test[i][0] for i in range(len(dataset_test))], dim=0)),
                dim=0)
            self.targets: Tensor = torch.cat((Tensor(dataset_train.targets), Tensor(dataset_test.targets)), dim=0) \
                .type(torch.int64)
            torch.save({'x': self.data, 'y': self.targets, 'classes': dataset_test.classes},
                       self.PATH_TRANSFORMED.format('100' if cifar100 else '10'))
            self.classes = dataset_test.classes

        self.len: int = self.targets.shape[0]
        self.num_classes = len(self.classes)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return self.len

    def to(self, device: Union[torch.device, str]) -> None:
        self.data = self.data.to(device)
        self.targets = self.targets.to(device)
