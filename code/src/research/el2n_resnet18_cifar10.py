import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
import torchvision.utils
from torch import tensor

from code.src.prune.el2n import get_prune_idx, get_el2n_scores
from code.src.utils.common import get_model_resnet18_cifar10, get_loader, create_saved_data_dir, get_device, load, save
from code.src.utils.dataset import get_cifar10
from code.src.utils.plot import plot_prune_example, compare_models_losses
from code.src.utils.train import train

NUM_CLASSES = 10
BATCH_SIZE = 5
NUM_TRAIN = 30
NUM_VALID = 20
NUM_TEST = 20
EPOCHS = 1
ENSEMBLE_SIZE = 1

DEVICE = get_device()
PATH_MODELS_SAVE = create_saved_data_dir(__file__)


def main():
    torch.manual_seed(5)

    # get data
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    data_train, data_test = get_cifar10(os.path.abspath(os.path.join('../../../', 'datasets')))

    loader_train = get_loader(data_train, np.arange(NUM_TRAIN), BATCH_SIZE)
    loader_valid = get_loader(data_train, np.arange(NUM_TRAIN, NUM_VALID + NUM_TRAIN), BATCH_SIZE)
    loader_test = get_loader(data_test, np.arange(NUM_TEST), BATCH_SIZE)

    idx = np.arange(NUM_TRAIN)
    # create loader with no shuffling
    loader_train_ordered = get_loader(data_train, idx, BATCH_SIZE, shuffle=False)
    Y_train = tensor(data_train.targets)[idx]

    ensemble = [get_model_resnet18_cifar10() for _ in range(ENSEMBLE_SIZE)]
    ensemble_softmax = torch.empty((len(ensemble), NUM_TRAIN, NUM_CLASSES))
    ensemble_pred = torch.empty((NUM_TRAIN, len(ensemble)), dtype=torch.bool)

    print()
    for i, (model, criterion, optimizer, scheduler) in enumerate(ensemble):
        print(f'------------   model {i}   -------------------')
        path = PATH_MODELS_SAVE(f'resnet18_{i}')
        train(model, loader_train, loader_valid, loader_test, criterion, optimizer, scheduler, 2, NUM_CLASSES, DEVICE,
              verbose=True, save_path=path)

        model.eval()
        for batch_idx, (X, y) in enumerate(loader_train_ordered):
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            idx = np.arange(pred.shape[0]) * batch_idx
            ensemble_softmax[i, idx] = functional.softmax(pred, dim=1).clone().detach().cpu()
            ensemble_pred[idx, i] = torch.Tensor(torch.max(pred, 1)[1].type(torch.int8) == y).clone().detach().cpu()

    ensemble_pred_sum = torch.sum(ensemble_pred, dim=1)
    ensemble_var = ensemble_softmax.var(dim=0)

    plt.style.use('ggplot')
    plt.hist(ensemble_pred_sum, bins=len(ensemble), facecolor='g', alpha=0.6)
    plt.show()

    el2n_scores = get_el2n_scores(Y_train, ensemble_softmax)
    plt.hist(el2n_scores, bins=len(data_train.classes), facecolor='g', alpha=0.6)
    plt.show()

    data_train_raw = torchvision.datasets.CIFAR10(os.path.abspath(r'../../../datasets'), train=True)
    plot_prune_example(data_train_raw, el2n_scores, hardest=True, prune_method_name='EL2N', random=False)
    plot_prune_example(data_train_raw, el2n_scores, hardest=False, prune_method_name='EL2N', random=False)

    idx_to_keep = get_prune_idx(Y_train, ensemble_softmax, .5)

    # train model with prune
    print("\nrun model with prune")
    loader_train_prune = get_loader(data_train, idx_to_keep, BATCH_SIZE, True)
    model_prune, criterion_prune, optimizer_prune, scheduler_prune = get_model_resnet18_cifar10()
    res_train_prune, res_valid_prune, _ = \
        train(model_prune, loader_train_prune, loader_valid, loader_test, criterion_prune, optimizer_prune,
              scheduler_prune, EPOCHS, NUM_CLASSES, DEVICE, verbose=True, save_path=PATH_MODELS_SAVE('resnet18_prune'))
    scores_train_prune, pred_train_prune, loss_train_prune, acc_train_prune = res_train_prune
    scores_valid_prune, pred_valid_prune, loss_valid_prune, acc_valid_prune = res_valid_prune

    # save data
    save(PATH_MODELS_SAVE(''), ensemble_pred=ensemble_pred, ensemble_pred_sum=ensemble_pred_sum,
         ensemble_softmax=ensemble_softmax, ensemble_var=ensemble_var, el2n_scores=el2n_scores,
         idx_to_keep=idx_to_keep, scores_train_prune=scores_train_prune, pred_train_prune=pred_train_prune,
         loss_train_prune=loss_train_prune, acc_train_prune=acc_train_prune, scores_valid_prune=scores_valid_prune,
         pred_valid_prune=pred_valid_prune, loss_valid_prune=loss_valid_prune, acc_valid_prune=acc_valid_prune)


if __name__ == '__main__':
    main()
