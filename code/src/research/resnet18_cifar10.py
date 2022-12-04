import os

import numpy as np
import torch
from torch import tensor

from code.src.utils.common import get_model_resnet18_cifar10, get_loader, create_saved_data_dir, get_device
from code.src.utils.dataset import get_cifar10
from code.src.utils.train import train

NUM_CLASSES = 10
BATCH_SIZE = 5
NUM_TRAIN = 30
NUM_VALID = 20
NUM_TEST = 20
EPOCHS = 121

DEVICE = get_device()
PATH_MODELS_SAVE = create_saved_data_dir(__file__)


def main():
    torch.manual_seed(5)

    # get data
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    data_train, data_test = get_cifar10(os.path.abspath(os.path.join('../../../', 'datasets')))
    Y_train = tensor(data_train.targets)[np.arange(NUM_TRAIN)]

    loader_train = get_loader(data_train, np.arange(NUM_TRAIN), BATCH_SIZE)
    loader_valid = get_loader(data_train, np.arange(NUM_TRAIN, NUM_VALID + NUM_TRAIN), BATCH_SIZE)
    loader_test = get_loader(data_test, np.arange(NUM_TEST), BATCH_SIZE)

    # train model without prune
    print("\nrun model without prune")
    model_simple, criterion_simple, optimizer_simple, scheduler_simple = get_model_resnet18_cifar10()
    res_train, res_valid, _ = train(model_simple, loader_train, loader_valid, loader_test, criterion_simple,
                                    optimizer_simple, scheduler_simple, EPOCHS, NUM_CLASSES, DEVICE, verbose=True,
                                    save_path=PATH_MODELS_SAVE('resnet18_no_prune'))
    scores_train, pred_train, loss_train, acc_train = res_train
    scores_valid, pred_valid, loss_valid, acc_valid = res_valid


if __name__ == '__main__':
    main()
