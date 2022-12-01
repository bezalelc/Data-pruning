from enum import Enum
import numpy as np
import torch


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
