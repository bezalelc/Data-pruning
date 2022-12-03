import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils
from torch import tensor

from code.src.prune.el2n import get_prune_idx, get_el2n_scores
from code.src.utils.dataset import get_cifar10
from code.src.utils.train import train
from code.src.utils.common import get_model_resnet18_cifar10,get_loader,create_saved_data_dir,get_device
from code.src.utils.plot import plot_prune_example


NUM_CLASSES = 10
BATCH_SIZE = 20
NUM_TRAIN = 1000
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

    loader_train = get_loader(data_train, np.arange(NUM_TRAIN),BATCH_SIZE)
    loader_valid = get_loader(data_train, np.arange(NUM_TRAIN, NUM_VALID + NUM_TRAIN),BATCH_SIZE)
    loader_test = get_loader(data_test, np.arange(NUM_TEST),BATCH_SIZE)

    ensemble = [get_model_resnet18_cifar10() for _ in range(ENSEMBLE_SIZE)]
    prune_size = .5
    ensemble_softmax = torch.empty((len(ensemble), NUM_TRAIN, NUM_CLASSES))
    ensemble_pred = torch.empty((NUM_TRAIN, len(ensemble)), dtype=torch.bool)
    idx = np.arange(NUM_TRAIN)
    # create loader with no shuffling
    loader_prune = get_loader(data_train, idx,BATCH_SIZE, shuffle=False)
    Y_train = tensor(data_train.targets)[idx]

    for i, (model, criterion, optimizer) in enumerate(ensemble):
        print(f'------------   model {i}   -------------------')
        path = os.path.join(PATH_MODELS_SAVE, f'resnet18_{i}')
        train(model, loader_train, loader_valid, loader_test, criterion, optimizer, 2, NUM_CLASSES,DEVICE, verbose=True, save_path=path)

        model.eval()
        for batch_idx, (X, y) in enumerate(loader_prune):
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            idx = np.arange(pred.shape[0]) * batch_idx
            ensemble_softmax[i, idx] = F.softmax(pred, dim=1).clone().detach().cpu()
            ensemble_pred[idx, i] = (torch.max(pred, 1)[1].type(torch.int8) == y).clone().detach().cpu()

    ensemble_pred_sum = torch.sum(ensemble_pred, dim=1)
    ensemble_var = ensemble_softmax.var(dim=0)

    # save data
    os.makedirs(PATH_MODELS_SAVE, exist_ok=True)
    torch.save(ensemble_pred_sum, os.path.join(PATH_MODELS_SAVE, 'ensemble_pred_sum.pt'))
    torch.save(ensemble_pred, os.path.join(PATH_MODELS_SAVE, 'ensemble_pred.pt'))
    torch.save(ensemble_softmax, os.path.join(PATH_MODELS_SAVE, 'ensemble_softmax.pt'))
    torch.save(ensemble_var, os.path.join(PATH_MODELS_SAVE, 'ensemble_var.pt'))

    #plt.style.use('ggplot')
    #plt.hist(ensemble_pred_sum, bins=len(ensemble), facecolor='g', alpha=0.6)
    #plt.show()

    el2n_scores = get_el2n_scores(Y_train, ensemble_softmax)
    #plt.hist(el2n_scores, bins=len(data_train.classes), facecolor='g', alpha=0.6)
    #plt.show()

    data_train_raw = torchvision.datasets.CIFAR10(os.path.abspath(r'../../../datasets'), train=True)
    plot_prune_example(data_train_raw,el2n_scores,hardest=True,prune_method_name='EL2N')
    plot_prune_example(data_train_raw,el2n_scores,hardest=False,prune_method_name='EL2N')

    idx_to_keep = get_prune_idx(Y_train, ensemble_softmax, prune_size)

    loader_train = get_loader(data_train, idx_to_keep,BATCH_SIZE, True)
    model_prune, criterion_prune, optimizer_prune = get_model_resnet18_cifar10()
    model_simple, criterion_simple, optimizer_simple = get_model_resnet18_cifar10()

    # train model with prune
    (scores_train_p, pred_train_p, loss_train_p, acc_train_p), \
        (scores_valid_p, pred_valid_p, loss_valid_p, acc_valid_p), res_test = \
        train(model_prune, loader_prune, loader_valid, loader_test, criterion_prune, optimizer_prune, EPOCHS,
              NUM_CLASSES, DEVICE,              verbose=True, save_path=os.path.join(PATH_MODELS_SAVE, 'resnet18_prune'))

    # train model without prune
    (scores_train, pred_train, loss_train, acc_train), \
        (scores_valid, pred_valid, loss_valid, acc_valid), res_test = \
        train(model_simple, loader_train, loader_valid, loader_test, criterion_simple, optimizer_simple, EPOCHS,
              NUM_CLASSES,DEVICE,verbose=True, save_path=os.path.join(PATH_MODELS_SAVE, 'resnet18_no_prune'))


    torch.save(model_prune.state_dict(), './model_prune.pt')
    torch.save(model_simple.state_dict(), './model_simple.pt')
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

    # plt.show()


if __name__ == '__main__':
    main()
