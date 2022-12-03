import torch
from torch import tensor


def get_el2n_scores(y:tensor, ensemble_pred:tensor):
    """
    calculate mean on the L2 over ensemble of algorithms

    :param y: labels, shape: (data len)
    :param ensemble_pred: scores for every data example, shape: (ensemble size, data len, labels len)

    :return: el2n_scores: vector of scores how the example hard to learn for every data
             shape: (data len)
    """
    y_one_hot = torch.nn.functional.one_hot(y, num_classes=ensemble_pred.shape[-1])
    return torch.mean(torch.linalg.norm(y_one_hot - ensemble_pred, ord=2, dim=2), dim=0)


def get_el2n_scores_(y:tensor, ensemble_pred:tensor):
    """
    calculate L2(mean on the scores) over ensemble of algorithms

    :param y: labels, shape: (data len)
    :param ensemble_pred: scores for every data example, shape: (ensemble size, data len, labels len)

    :return: el2n_scores: vector of scores how the example hard to learn for every data
             shape: (data len)
    """
    mean_ensemble_pred = torch.mean(ensemble_pred, dim=0)
    y_ = torch.nn.functional.one_hot(y, num_classes=ensemble_pred.shape[-1])
    return torch.linalg.norm(y_ - mean_ensemble_pred, 2, 1)


def get_prune_idx(y:tensor, ensemble_pred:tensor, prune_size: float, keep_hardest: bool = True):
    return get_el2n_scores(y, ensemble_pred).argsort(descending=keep_hardest)[:int(prune_size * y.shape[0])]
