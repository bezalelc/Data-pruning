import torch
import pytest
from torch import tensor

import code.src.prune.el2n as el2n


@pytest.fixture
def dataset():
    return torch.arange(-24, 0, dtype=torch.float64).reshape((2, 3, 4)), torch.arange(3)


def test_get_el2n_scores(dataset):
    from torch import tensor

    ensemble_softmax = tensor([
        [[1, 0]],
        [[0, 1]]
    ], dtype=torch.float64)
    num_ensemble, num_train, num_classes = ensemble_softmax.shape
    Y_train = tensor([0])
    y_one_hot_ = tensor([[1, 0]])
    expected = (y_one_hot_ - ensemble_softmax) ** 2
    expected = tensor([(expected[0][0][0] + expected[0][0][1]) ** .5,
                       (expected[1][0][0] + expected[1][0][1]) ** .5])
    expected = (expected[0] + expected[1]) / num_ensemble
    el2n_scores = el2n.get_el2n_scores(Y_train, ensemble_softmax)
    assert expected == el2n_scores


def test_get_prune_idx(dataset):
    X, Y = dataset

    idx_to_keep = el2n.get_prune_idx(Y, X, .5)
    assert idx_to_keep.shape[0] == 12
