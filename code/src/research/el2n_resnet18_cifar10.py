import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
import torchvision.utils
from torch import tensor

from code.src.prune.el2n import get_prune_idx, get_el2n_scores
from code.src.utils.common import get_model_resnet18_cifar10, get_loader, create_saved_data_dir, get_device
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
    data_train, data_test = get_cifar10()

    loader_train = get_loader(data_train, np.arange(NUM_TRAIN), BATCH_SIZE)
    loader_valid = get_loader(data_train, np.arange(NUM_TRAIN, NUM_VALID + NUM_TRAIN), BATCH_SIZE)
    loader_test = get_loader(data_test, np.arange(NUM_TEST), BATCH_SIZE)

    ensemble = [get_model_resnet18_cifar10() for _ in range(ENSEMBLE_SIZE)]
    prune_size = .5
    ensemble_softmax = torch.empty((len(ensemble), NUM_TRAIN, NUM_CLASSES))
    ensemble_pred = torch.empty((NUM_TRAIN, len(ensemble)), dtype=torch.bool)
    idx = np.arange(NUM_TRAIN)
    # create loader with no shuffling
    loader_train_ordered = get_loader(data_train, idx, BATCH_SIZE, shuffle=False)
    Y_train = tensor(data_train.targets)[idx]

    print()
    for i, (model, criterion, optimizer) in enumerate(ensemble):
        print(f'------------   model {i}   -------------------')
        path = PATH_MODELS_SAVE(f'resnet18_{i}')
        train(model, loader_train, loader_valid, loader_test, criterion, optimizer, 2, NUM_CLASSES, DEVICE,
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

    # save data
    torch.save(ensemble_pred_sum, PATH_MODELS_SAVE('ensemble_pred_sum.pt'))
    torch.save(ensemble_pred, PATH_MODELS_SAVE('ensemble_pred.pt'))
    torch.save(ensemble_softmax, PATH_MODELS_SAVE('ensemble_softmax.pt'))
    torch.save(ensemble_var, PATH_MODELS_SAVE('ensemble_var.pt'))

    plt.style.use('ggplot')
    plt.hist(ensemble_pred_sum, bins=len(ensemble), facecolor='g', alpha=0.6)
    plt.show()

    el2n_scores = get_el2n_scores(Y_train, ensemble_softmax)
    plt.hist(el2n_scores, bins=len(data_train.classes), facecolor='g', alpha=0.6)
    plt.show()

    data_train_raw = torchvision.datasets.CIFAR10(os.path.abspath(r'../../../datasets'), train=True)
    plot_prune_example(data_train_raw, el2n_scores, hardest=True, prune_method_name='EL2N', random=False)
    plot_prune_example(data_train_raw, el2n_scores, hardest=False, prune_method_name='EL2N', random=False)

    idx_to_keep = get_prune_idx(Y_train, ensemble_softmax, prune_size)

    # train model with prune
    print("\nrun model with prune")
    loader_train_prune = get_loader(data_train, idx_to_keep, BATCH_SIZE, True)
    model_prune, criterion_prune, optimizer_prune = get_model_resnet18_cifar10()
    res_train_p, res_valid_p, _ = \
        train(model_prune, loader_train_prune, loader_valid, loader_test, criterion_prune, optimizer_prune, EPOCHS,
              NUM_CLASSES, DEVICE, verbose=True, save_path=PATH_MODELS_SAVE('resnet18_prune'))
    scores_train_p, pred_train_p, loss_train_p, acc_train_p = res_train_p
    scores_valid_p, pred_valid_p, loss_valid_p, acc_valid_p = res_valid_p

    # train model without prune
    print("\nrun model without prune")
    model_simple, criterion_simple, optimizer_simple = get_model_resnet18_cifar10()
    res_train, res_valid, _ = train(model_simple, loader_train, loader_valid, loader_test, criterion_simple,
                                    optimizer_simple, EPOCHS, NUM_CLASSES, DEVICE, verbose=True,
                                    save_path=PATH_MODELS_SAVE('resnet18_no_prune'))
    scores_train, pred_train, loss_train, acc_train = res_train
    scores_valid, pred_valid, loss_valid, acc_valid = res_valid

    # load: model.load_state_dict(torch.load(PATH))

    compare_models_losses(loss_train, loss_train_p, loss_valid, loss_valid_p)


if __name__ == '__main__':
    main()
