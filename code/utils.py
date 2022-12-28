"""
helpers for common datasets operations
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

# from torch.utils.data.dataloader import default_collate

PATH_ROOT = os.path.abspath(os.path.join(os.path.split(__file__)[0], '../'))
PATH_DATASETS = os.path.join(PATH_ROOT, 'datasets')

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


def get_cifar100() -> \
        tuple[
            torchvision.datasets.CIFAR100,
            torchvision.datasets.CIFAR100,
            torchvision.datasets.CIFAR100,
            torchvision.datasets.CIFAR100
        ]:
    """
    get cifar100 train set and test set
    """
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
    dataset_train_for_test = torchvision.datasets.CIFAR100(PATH_DATASETS, train=True, download=False,
                                                           transform=transform_test)
    dataset_train_raw = torchvision.datasets.CIFAR100(PATH_DATASETS, train=True, download=False)

    return dataset_train, dataset_test, dataset_train_for_test, dataset_train_raw


def get_loader(dataset: torchvision.datasets, idx, batch_size: int,
               shuffle: bool = True) -> torch.utils.data.DataLoader:
    """
    get data loader according to indexes

    Args:
        dataset:
        idx:
        batch_size:
        shuffle:

    Returns: torch.utils.data.DataLoader

    """
    subset = torch.utils.data.Subset(dataset, idx)
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=4,
                                         prefetch_factor=4, pin_memory=True, persistent_workers=True)

    return loader


def plot_img_and_top(dataset: torch.utils.data.DataLoader, range_: tuple, scores: Tensor,
                     softmax: Tensor, ensemble: bool = False, score_name: str = ''):
    num_train = len(scores)
    num_plots, num_top = 5, 12
    classes = dataset.classes
    idx = np.random.randint(int(num_train * range_[0]), int(num_train * range_[1]), num_plots)
    idx = scores.sort()[1].numpy()[idx]

    if ensemble:
        softmax = softmax.mean(dim=0)

    plt.style.use('default')
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))

    for i, ax_txt, ax_img in zip(idx, axes[0], axes[1]):
        s = f'{score_name} score: {scores[i]:.3f}\nTrue class: {classes[dataset[i][1]]}'  # score: {scores[i]}\n
        top_scores, top_classes_idx = softmax[i].sort(descending=True)
        for score, j in zip(top_scores[:num_top], top_classes_idx[:num_top]):
            s += f"\n{(classes[j] + ':'):<15} {score:.0%}"

        ax_txt.text(.0, .0, s, dict(size=14, family='monospace'))
        ax_txt.axis('off')
        ax_img.imshow(dataset[i][0])

    plt.show()

# def get_el2n_scores(y: Tensor, ensemble_pred: Tensor):
#     """
#     calculate mean on the L2 over ensemble of algorithms
#
#     :param y: labels, shape: (data len)
#     :param ensemble_pred: scores for every data example, shape: (ensemble size, data len, labels len)
#
#     :return: el2n_scores: vector of scores how the example hard to learn for every data
#              shape: (data len)
#     """
#     y_one_hot = torch.nn.functional.one_hot(y, num_classes=ensemble_pred.shape[-1])
#     return torch.mean(torch.linalg.norm(y_one_hot - ensemble_pred, ord=2, dim=2), dim=0)
#
