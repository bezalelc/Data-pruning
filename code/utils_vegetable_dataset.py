"""
helpers for common datasets operations for vegetable dataset
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import Tensor
from torch.utils.data import Dataset


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
    num_plots, num_top = 5, 10
    classes = dataset.classes
    idx = np.random.randint(int(num_train * range_[0]), int(num_train * range_[1]), num_plots)
    idx = scores.sort()[1].numpy()[idx]

    if ensemble:
        softmax = softmax.mean(dim=0)

    plt.style.use('default')
    fig, axes = plt.subplots(2, 5, figsize=(18, 4))

    for i, ax_txt, ax_img in zip(idx, axes[0], axes[1]):
        s = f'{score_name} score: {scores[i]:.3f}\nTrue class: {classes[dataset[i][1]]}'  # score: {scores[i]}\n
        top_scores, top_classes_idx = softmax[i].sort(descending=True)
        for score, j in zip(top_scores[:num_top], top_classes_idx[:num_top]):
            s += f"\n{(classes[j] + ':'):<15} {score:.0%}"

        ax_txt.text(.0, .0, s, dict(size=14, family='monospace'))
        ax_txt.axis('off')
        ax_img.imshow(dataset[i][0])

    plt.show()
