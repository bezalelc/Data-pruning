import torch
import pytest
import code.src.prune.el2n as el2n


@pytest.fixture
def dataset():
    return torch.arange(-24, 0, dtype=torch.float64).reshape((2, 3, 4)), torch.arange(3)


def test_get_el2n_scores(dataset):
    X, Y = dataset

    X_mean = torch.tensor([[6., 7., 8., 9.],
                           [10., 11., 12., 13.],
                           [14., 15., 16., 17.]], dtype=torch.float64)
    res = X_mean.detach()
    res[0, Y[0]] -= 1
    res[1, Y[1]] -= 1
    res[2, Y[2]] -= 1

    res = torch.sum(res * res, dim=1) ** 0.5
    el2n_scores = el2n.get_el2n_scores(Y, X)


def test_get_prune_idx(dataset):
    X, Y = dataset

    idx_to_keep = el2n.get_prune_idx(Y, X, .5)
    assert idx_to_keep.shape[0] == 12
