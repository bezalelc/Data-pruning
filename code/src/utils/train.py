from enum import Enum

import numpy as np
import torch
from torch import nn


class Mode(Enum):
    TRAIN = 0
    VALIDATE = 1
    TEST = 2


def run_epoch(model, criterion, optimizer, loader, num_classes, device, mode: Mode = Mode.TRAIN):
    model.train() if mode == Mode.TRAIN else model.eval()
    model.to(device)

    loss, loss_min, acc = .0, np.Inf, .0
    len_dataset = len(loader.dataset)
    scores = torch.empty((len(loader.dataset), num_classes))
    pred = torch.empty((len(loader.dataset),))

    for batch_idx, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        if mode == Mode.TRAIN:
            optimizer.zero_grad()

        p = model(X)
        loss_batch = criterion(p, y)
        loss += loss_batch.item()

        if mode == Mode.TRAIN:
            loss_batch.backward()
            optimizer.step()
        else:
            scores[batch_idx * loader.batch_size:(batch_idx + 1) * loader.batch_size] = p.clone().detach()

        _, pred_ = torch.max(p, 1)
        # print(batch_idx, y, pred, pred.eq(y), torch.sum(pred.eq(y)), torch.sum(pred.eq(y)) / 30)
        acc += torch.sum(pred_.eq(y))
        pred[batch_idx * loader.batch_size:(batch_idx + 1) * loader.batch_size] = pred_

    return scores, pred, loss / len_dataset, acc / len_dataset


def train(model: nn.Module, train_loader: torch.utils.data.dataloader.DataLoader, valid_loader, test_loader, criterion,
          optimizer: torch.optim.Optimizer, scheduler, epochs: int,
          num_classes: int, device: torch.device, save_path='', verbose: bool = True):
    loss_train, loss_valid, loss_valid_min, acc_train, acc_valid = [], [], np.Inf, [], []
    scores_train, scores_valid, pred_train, pred_valid = None, None, None, None

    for epoch in range(epochs):
        scores_train, pred_train, loss, acc = run_epoch(model, criterion, optimizer, train_loader, num_classes,
                                                        device, Mode.TRAIN)
        loss_train.append(loss), acc_train.append(acc)
        scores_valid, pred_valid, loss, acc = run_epoch(model, criterion, optimizer, valid_loader, num_classes,
                                                        device, Mode.VALIDATE)
        loss_valid.append(loss), acc_valid.append(acc)
        scheduler.step()

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

    scores_test, pred_test, loss_test, acc_test = run_epoch(model, criterion, optimizer, test_loader, num_classes,
                                                            device, Mode.TEST)
    if verbose:
        print(f'Test Loss: {loss_test:.6f}')
        print(f'Accuracy: {acc_test}')

    return (scores_train, pred_train, loss_train, acc_train), (scores_valid, pred_valid, loss_valid, acc_valid), \
           (scores_test, pred_test, loss_test, acc_test)
