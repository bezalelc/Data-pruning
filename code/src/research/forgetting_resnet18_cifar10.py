import os
import matplotlib.pyplot as plt

import numpy as np
import torch
import torchvision

from code.src.utils.dataset import get_cifar10
from code.src.utils.train import Mode, run_epoch
from code.src.utils.common import get_model_resnet18_cifar10,get_loader,create_saved_data_dir,get_device
from code.src.utils.plot import plot_prune_example

NUM_CLASSES = 10
BATCH_SIZE = 5
NUM_TRAIN = 100
NUM_VALID = 10
NUM_TEST = 5
EPOCHS = 10

DEVICE = get_device()
PATH_MODELS_SAVE = create_saved_data_dir(__file__)
torch.manual_seed(5)



def main():
    # get data
    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    data_train, data_test = get_cifar10()
    loader_train = get_loader(data_train, np.arange(NUM_TRAIN),BATCH_SIZE)
    loader_valid = get_loader(data_train, np.arange(NUM_TRAIN, NUM_VALID + NUM_TRAIN),BATCH_SIZE)
    loader_test = get_loader(data_test, np.arange(NUM_TEST),BATCH_SIZE)

    model, criterion, optimizer = get_model_resnet18_cifar10()
    epochs_pred = torch.empty((EPOCHS, NUM_TRAIN))

    for epoch in range(EPOCHS):
        train_res = run_epoch(model, criterion, optimizer, loader_train, NUM_CLASSES, DEVICE, Mode.TRAIN)
        valid_res = run_epoch(model, criterion, optimizer, loader_valid, NUM_CLASSES, DEVICE, Mode.VALIDATE)
        scores_train, pred_train, loss_train, acc_train=train_res
        scores_valid, pred_valid, loss_valid, acc_valid=valid_res
        print(f'Epoch: {epoch} Training: Loss: {loss_train:.6f} Acc: {acc_train:.6f}  '
              f'Validation Loss: {loss_valid:.6f} Acc: {acc_valid:.6f}')
        epochs_pred[epoch] = pred_train.detach().clone()

    change_counter = torch.zeros(NUM_TRAIN, dtype=torch.int8)
    for p1, p2 in zip(epochs_pred[:-1], epochs_pred[1:]):
        change_counter += p1 != p2

    os.makedirs(PATH_MODELS_SAVE, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(PATH_MODELS_SAVE, 'model.pt'))
    torch.save(epochs_pred, os.path.join(PATH_MODELS_SAVE, 'epochs_pred.pt'))
    torch.save(change_counter, os.path.join(PATH_MODELS_SAVE, 'change_counter.pt'))

    #plt.hist(change_counter, bins=EPOCHS - 1, facecolor='g', alpha=0.6)
    #plt.show()
    data_train_raw = torchvision.datasets.CIFAR10(os.path.abspath(r'../../../datasets'), train=True)
    plot_prune_example(data_train_raw, change_counter, hardest=True)
    plot_prune_example(data_train_raw, change_counter.type(torch.float), hardest=False)



if __name__ == '__main__':
    main()
