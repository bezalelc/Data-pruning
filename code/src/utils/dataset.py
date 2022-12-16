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


def get_cifar(path_dataset: str, cifar100: bool = True) -> \
        tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
    """
    get cifar10 train set and test set

    Returns:
        cifar10 train set and test set
    """
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if cifar100:
        transform = transforms.Compose([
            # transforms.Pad(4),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32),
            transforms.ToTensor(),
            # transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047))
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        # [0.50707516 0.48654887 0.44091784] [0.26733429 0.25643846 0.27615047]
        # [0.50736203 0.48668956 0.44108857] [0.26748815 0.2565931  0.27630851] with train
        dataset_train = torchvision.datasets.CIFAR100(path_dataset, train=True, transform=transform, download=True)
        dataset_test = torchvision.datasets.CIFAR100(path_dataset, train=False, transform=transform, download=True)
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

    return dataset_train, dataset_test


class GPUDataset(Dataset):
    PATH_DATASETS: str = os.path.abspath(os.path.join('../../../', 'datasets'))
    PATH_TRANSFORMED: str = os.path.abspath(os.path.join(PATH_DATASETS, 'gpu', 'cifar{}_transformed.pt'))
    NORM_100 = ((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047))
    NORM_10 = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    NORM_05 = ((.5, .5, .5), (.5, .5, .5))

    def __init__(self, load: bool = False, cifar100: bool = True) -> None:
        super().__init__()

        if load:
            dataset = torch.load(
                self.PATH_TRANSFORMED.format('100' if cifar100 else '10').format('100' if cifar100 else '10'))
            self.data, self.targets, self.classes = dataset['x'], dataset['y'], dataset['classes']
        else:
            transform = transforms.Compose([
                # transforms.Pad(4),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((.5, .5, .5), (.5, .5, .5))
                # transforms.Normalize(*self.NORM_05)
            ])
            if cifar100:
                dataset_train = torchvision.datasets.CIFAR100(self.PATH_DATASETS, train=True, transform=transform,
                                                              download=True)
                dataset_test = torchvision.datasets.CIFAR100(self.PATH_DATASETS, train=False, transform=transform,
                                                             download=True)
            else:
                dataset_train = torchvision.datasets.CIFAR10(self.PATH_DATASETS, train=True, transform=transform,
                                                             download=True)
                dataset_test = torchvision.datasets.CIFAR10(self.PATH_DATASETS, train=False, transform=transform,
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
