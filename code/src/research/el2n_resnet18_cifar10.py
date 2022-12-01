import os
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils
from torch import nn, optim, tensor
from torchvision import models

from code.src.prune.el2n import get_prune_idx, get_el2n_scores
from code.src.utils.dataset import get_cifar10
from code.src.utils.train import Mode, run_epoch

NUM_CLASSES = 10
BATCH_SIZE = 5
NUM_TRAIN = 50
NUM_VALID = 5
NUM_TEST = 5
EPOCHS = 2
ENSEMBLE_SIZE = 3
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


def train(model, train_loader, valid_loader, test_loader, criterion, optimizer, epochs: int, save_path='',
          verbose: bool = True):
    loss_train, loss_valid, loss_valid_min, acc_train, acc_valid = [], [], np.Inf, [], []
    scores_train, scores_valid, scores_test = None, None, None

    for epoch in range(epochs):
        scores_train, pred_train, loss, acc = run_epoch(model, criterion, optimizer, train_loader, NUM_CLASSES,
                                                        Mode.TRAIN, TRAIN_ON_GPU)
        loss_train.append(loss), acc_train.append(acc)
        scores_valid, pred_valid, loss, acc_test = run_epoch(model, criterion, optimizer, valid_loader, NUM_CLASSES,
                                                             Mode.VALIDATE, TRAIN_ON_GPU)
        loss_valid.append(loss), acc_valid.append(acc)

        # print training/validation statistics
        if verbose:
            print(f'Epoch: {epoch} Training: Loss: {loss_train[-1]:.6f} Acc: {acc_train[-1]:.6f}  '
                  f'Validation Loss: {loss_valid[-1]:.6f} Acc: {acc_valid[-1]:.6f}')

        # save model if validation loss has decreased
        if save_path and loss_valid[-1] <= loss_valid_min:
            if verbose:
                print(f'Validation loss decreased ({loss_valid_min:.6f} --> {loss_valid[-1]:.6f}).  '
                      f'Saving model to {save_path}')
            torch.save(model.state_dict(), save_path)
            loss_valid_min = loss_valid[-1]

    scores_test, pred_test, loss_test, acc_test = run_epoch(model, criterion, optimizer, test_loader, NUM_CLASSES,
                                                            Mode.TEST, TRAIN_ON_GPU)
    if verbose:
        print(f'Test Loss: {loss_test:.6f}')
        print(f'Accuracy: {acc_test}')

    return (scores_train, loss_train, acc_train), (scores_valid, loss_valid, acc_valid), \
           (scores_test, loss_test, acc_test)


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

    ensemble = [get_model() for _ in range(ENSEMBLE_SIZE)]
    prune_size = .5
    ensemble_softmax = torch.empty((len(ensemble), NUM_TRAIN, NUM_CLASSES), device=DEVICE)
    ensemble_pred = torch.empty((NUM_TRAIN, len(ensemble)), dtype=torch.bool, device=DEVICE)
    ensemble_pred_sum = torch.empty((NUM_TRAIN,), dtype=torch.int8, device=DEVICE)
    idx = np.arange(NUM_TRAIN)
    # create loader with no shuffling
    loader_prune = get_loader(data_train, idx, shuffle=False)
    Y_train = tensor(data_train.targets, device=DEVICE)[idx]

    for i, (model, criterion, optimizer) in enumerate(ensemble):
        print(f'------------   model {i}   -------------------')
        path = os.path.join(PATH_MODELS_SAVE, f'resnet18_{i}')
        (scores_train, loss_train, acc_train), (scores_valid, loss_valid, acc_valid), (
            scores_test, loss_test, acc_test) = \
            train(model, loader_train, loader_valid, loader_test, criterion, optimizer, 4, verbose=True, save_path=path)

        model.eval()
        for batch_idx, (X, y) in enumerate(loader_prune):
            if TRAIN_ON_GPU:
                X, y = X.cuda(), y.cuda()
            pred = model(X)
            idx = np.arange(batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE)
            ensemble_softmax[i, idx] = F.softmax(pred, dim=1)
            ensemble_pred[idx, i] = torch.max(pred, 1)[1].type(torch.int8) == y

    ensemble_pred_sum = torch.sum(ensemble_pred, dim=1)
    ensemble_var = ensemble_softmax.var(dim=0)

    # save data
    torch.save(ensemble_pred_sum, os.path.join(PATH_MODELS_SAVE, 'ensemble_pred_sum.pt'))
    torch.save(ensemble_pred, os.path.join(PATH_MODELS_SAVE, 'ensemble_pred.pt'))
    torch.save(ensemble_softmax, os.path.join(PATH_MODELS_SAVE, 'ensemble_softmax.pt'))
    torch.save(ensemble_var, os.path.join(PATH_MODELS_SAVE, 'ensemble_var.pt'))

    plt.style.use('ggplot')
    plt.hist(ensemble_pred_sum, bins=len(ensemble), facecolor='g', alpha=0.6)
    plt.show()

    el2n_scores = get_el2n_scores(Y_train, ensemble_softmax).detach().numpy()
    plt.hist(el2n_scores, bins=len(data_train.classes), facecolor='g', alpha=0.6)
    plt.show()

    # [easy,...,hard]
    data_train_raw = torchvision.datasets.CIFAR10(os.path.abspath(r'../../../datasets'), train=True)
    el2n_scores_idx = np.argsort(el2n_scores)
    plt.style.use('default')
    fig, axes = plt.subplots(3, 4, figsize=(15, 15))
    fig.suptitle('Easiest examples')
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    for ax, i in zip(axes.reshape(-1), el2n_scores_idx[:12]):
        ax.imshow(data_train_raw[i][0])
        ax.set_title(f'EL2N {el2n_scores[i]:.3f}, Class: {data_train_raw.classes[data_train_raw[i][1]]}')
    plt.show()

    fig, axes = plt.subplots(3, 4, figsize=(15, 15))
    fig.suptitle('Hardest examples')
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    for ax, i in zip(axes.reshape(-1), el2n_scores_idx[-12:][::-1]):
        ax.imshow(data_train_raw[i][0])
        ax.set_title(f'EL2N {el2n_scores[i]:.3f}, Class: {data_train_raw.classes[data_train_raw[i][1]]}')
    plt.show()

    idx_to_keep = get_prune_idx(Y_train, ensemble_softmax, prune_size)

    loader_train = get_loader(data_train, idx_to_keep, True)
    model_prune, criterion_prune, optimizer_prune = get_model()
    model_simple, criterion_simple, optimizer_simple = get_model()

    # train model with prune
    (scores_train_p, loss_train_p, acc_train_p), (
        scores_valid_p, loss_valid_p, acc_valid_p), (
        scores_test_p, loss_test_p, acc_test_p) = \
        train(model_prune, loader_prune, loader_valid, loader_test, criterion_prune, optimizer_prune, EPOCHS,
              verbose=True, save_path=os.path.join(PATH_MODELS_SAVE, 'resnet18_prune'))

    # train model without prune
    (scores_train, loss_train, acc_train), (scores_valid, loss_valid, acc_valid), (
        scores_test, loss_test, acc_test) = \
        train(model_simple, loader_train, loader_valid, loader_test, criterion_simple, optimizer_simple, EPOCHS,
              verbose=True, save_path=os.path.join(PATH_MODELS_SAVE, 'resnet18_no_prune'))

    # torch.save(model_prune.state_dict(), './model_prune.pt')
    # torch.save(model_simple.state_dict(), './model_simple.pt')
    # load: model.load_state_dict(torch.load(PATH))

    fig, (ax_train_loss, ax_valid_loss) = plt.subplots(1, 2)
    ax_train_loss.plot(np.arange(EPOCHS), loss_train_p, label='prune')
    ax_train_loss.plot(np.arange(EPOCHS), loss_train, label='simple')
    ax_train_loss.set_title('train loss')
    ax_valid_loss.plot(np.arange(EPOCHS), loss_valid_p, label='prune')
    ax_valid_loss.plot(np.arange(EPOCHS), loss_valid, label='simple')
    ax_valid_loss.set_title('valid loss')

    for ax in (ax_train_loss, ax_valid_loss):
        ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.grid(True)
    ax.legend(loc='upper right')

    plt.show()


if __name__ == '__main__':
    main()
