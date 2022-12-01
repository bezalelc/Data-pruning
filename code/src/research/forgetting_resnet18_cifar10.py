import os
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn, optim
from torchvision import models

from code.src.utils.dataset import get_cifar10
from code.src.utils.train import Mode, run_epoch

NUM_CLASSES = 10
BATCH_SIZE = 5
NUM_TRAIN = 50
NUM_VALID = 5
NUM_TEST = 5
EPOCHS = 10
# PATH_MODELS_SAVE = r'/home/bb/Documents/Data-pruning/models_data/el2n_resnet18_cifar10'
dir_, f = os.path.split(__file__)
PATH_MODELS_SAVE = os.path.abspath(os.path.join(dir_, '../../../', 'models_data', f.split('.')[0]))
# check if CUDA is available
TRAIN_ON_GPU = torch.cuda.is_available()
DEVICE = 'cuda' if TRAIN_ON_GPU else 'cpu'


def get_loader(dataset, idx, shuffle=True):
    subset = torch.utils.data.Subset(dataset, idx)
    return torch.utils.data.DataLoader(subset, batch_size=BATCH_SIZE, shuffle=shuffle)


def get_model():
    model = models.resnet18(weights=None)  # ,pretrained=False
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.to(DEVICE)
    # model.load_state_dict(torch.load('ResNet18.pt'))

    # lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.0001)

    return model, criterion, optimizer


def main():
    if DEVICE == 'cuda':
        print('CUDA is available!  Training on  GPU...')
    else:
        print(f'CUDA is not available.  Training on {DEVICE.upper()}...')

    torch.manual_seed(5)

    # get data
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    data_train, data_test = get_cifar10()

    loader_train = get_loader(data_train, np.arange(NUM_TRAIN))
    loader_valid = get_loader(data_train, np.arange(NUM_TRAIN, NUM_VALID + NUM_TRAIN))
    loader_test = get_loader(data_test, np.arange(NUM_TEST))

    model, criterion, optimizer = get_model()
    epochs_pred = torch.empty((EPOCHS, NUM_TRAIN), device=DEVICE)

    for i in range(EPOCHS):
        print(f'------------   epoch {i}   -------------------')
        scores, pred, loss, acc = run_epoch(model, criterion, optimizer, loader_train, NUM_CLASSES, Mode.TRAIN,
                                            TRAIN_ON_GPU)
        print(f'Epoch: {i} Training: Loss: {loss:.6f} Acc: {acc:.6f}')
        epochs_pred[i] = pred

    change_counter = torch.zeros(NUM_TRAIN, device=DEVICE, dtype=torch.int8)
    for p1, p2 in zip(epochs_pred[:-1], epochs_pred[1:]):
        change_counter += p1 != p2

    os.makedirs(PATH_MODELS_SAVE, exist_ok=True)
    torch.save(epochs_pred, os.path.join(PATH_MODELS_SAVE, 'epochs_pred.pt'))
    torch.save(change_counter, os.path.join(PATH_MODELS_SAVE, 'change_counter.pt'))

    plt.hist(change_counter, bins=EPOCHS - 1, facecolor='g', alpha=0.6)
    plt.show()


if __name__ == '__main__':
    main()
