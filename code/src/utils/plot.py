import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Sequence,Union


def plot_prune_example(images_loader: torch.utils.data.DataLoader,
                       data_scores: torch.Tensor,
                       hardest: bool = True,
                       range_: float = .2,
                       random: bool = True,
                       prune_method_name: str = ''):
    """
    plot a bunch of examples

    Args:
        images_loader: data loader for images
        data_scores: scores for each example how much is hard to learn
        hardest: True/False -> plot the hardest/easiest
        range_ (float in range (0,1]): range to take random example from for example:
            range=.5 -> take random example from hardest/easiest examples
        random: plot random example from chosen range, if false plot the hardest/simplest examples
        prune_method_name: method of prune data
    """
    plot_num = 12
    choices = np.random.choice(int(range_ * data_scores.shape[0]), plot_num, replace=False) if random \
        else np.arange(plot_num)
    idx = data_scores.sort(descending=hardest)[1][choices]

    plt.style.use('default')
    fig, axes = plt.subplots(3, 4, figsize=(15, 15))
    fig.suptitle(f"{'Hardest' if hardest else 'Easiest'} examples")
    plt.subplots_adjust(right=.9, bottom=.1, top=.9)
    # plt.subplots_adjust(bottom=0.1, right=1., top=0.9)
    for ax, i in zip(axes.reshape(-1), idx):
        ax.imshow(images_loader[i][0])
        ax.set_title(f"{prune_method_name + ' '}{data_scores[i]:.3f}, "
                     f"Class: {images_loader.classes[images_loader[i][1]]}")
        # ax.set_facecolor('xkcd:salmon')
    plt.show()


def compare_models_losses(loss_train: list[float], loss_train_prune: list[float], loss_valid: list[float],
                          loss_valid_prune: list[float]):
    """
    Compare of train model on pruned data vs. not pruned data by plot the train/valid losses

    Args:
        loss_train:
        loss_valid:
        loss_train_prune:
        loss_valid_prune:
    """
    plt.style.use('ggplot')

    epochs = len(loss_train)
    fig, (ax_train_loss, ax_valid_loss) = plt.subplots(1, 2)
    ax_train_loss.plot(np.arange(epochs), loss_train_prune, label='prune')
    ax_train_loss.plot(np.arange(epochs), loss_train, label='simple')
    ax_train_loss.set_title('train loss')
    ax_valid_loss.plot(np.arange(epochs), loss_valid_prune, label='prune')
    ax_valid_loss.plot(np.arange(epochs), loss_valid, label='simple')
    ax_valid_loss.set_title('valid loss')

    for ax in (ax_train_loss, ax_valid_loss):
        ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.grid(True)
    ax.legend(loc='upper right')

    plt.show()


def plot_loss_acc(loss_train: Sequence, loss_valid: Sequence, acc_train: Sequence, acc_valid: Sequence):
    """
    plot result of training model progress

    Args:
        loss_train:
        loss_valid:
        acc_train:
        acc_valid:
    """
    assert len(loss_train) == len(loss_valid) == len(acc_train) == len(acc_valid)

    plt.style.use('ggplot')

    names = {0: 'loss', 1: 'acc'}
    epochs = np.arange(len(loss_train))
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.tight_layout()
    fig.subplots_adjust(wspace=.1)

    for i, ((tr, val), ax) in enumerate(zip(((loss_train, loss_valid), (acc_train, acc_valid)), axes)):
        ax.plot(epochs, tr, label='train')
        ax.plot(epochs, val, label='valid')
        ax.set_xlabel('epoch')
        ax.set_ylabel(names[i])
        ax.set_title(names[i])
        ax.grid(True)
        ax.legend(loc='upper right')

    plt.show()
