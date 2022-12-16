import os
import pickle
from typing import Callable, Union

import torch
import torchvision
from torch import nn, optim, Tensor
from torchvision import models

# class Loader(torch.utils.data.DataLoader):
#     def __init__(self, dataset, idx, batch_size: int, shuffle: bool = True):
#         super(Loader, self).__init(torch.utils.data.Subset(dataset, idx), batch_size=batch_size, shuffle=shuffle,
#                                    num_workers=4)
#         self.data = self.data.cuda()
from torch.utils.data.dataloader import default_collate


def get_loader(dataset: torchvision.datasets, idx, batch_size: int,
               shuffle: bool = True) -> torch.utils.data.DataLoader:
    """
    get data loader according to indexes

    Args:
        dataset:
        idx:
        batch_size:
        shuffle:

    Returns: torch.utils.data.DataLoader

    """
    subset = torch.utils.data.Subset(dataset, idx)
    # loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=4,
    #                                      collate_fn=lambda x: tuple(torch.tensor(x_).cuda() for x_ in default_collate(x)))
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=1, pin_memory=True)
    return loader


def get_model_resnet18_cifar10() -> tuple[
    torchvision.models.resnet.ResNet,
    torch.nn.modules.loss.CrossEntropyLoss,
    torch.optim.Adam,
    torch.optim.lr_scheduler.StepLR
]:
    """
    Returns: model, criterion, optimizer, scheduler

    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    return model, criterion, optimizer, scheduler


# print(len(get_model_resnet18_cifar10()[0].state_dict()))


def create_saved_data_dir(file: str) -> Callable[[str], str]:
    """
    create the path ../../../models_data/f if not exist and return function for get the relative path for saved data

    Args:
        file: file to save data in ../../../models_data/f

    Returns: function to generate path to file according to relative path to given file
    """
    dir_, f_ = os.path.split(file)
    path_to_save = os.path.abspath(os.path.join(dir_, '../../../', 'models_data', f_.split('.')[0]))
    os.makedirs(path_to_save, exist_ok=True)
    return lambda f=None: os.path.join(path_to_save, f) if f else path_to_save


def get_device() -> torch.device:
    """
    get device for pytorch: CPU or CUDA
    Returns: device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if str(device) == 'cuda':
        print('CUDA is available!  Training on  GPU...')
    else:
        print(f'CUDA is not available.  Training on {str(device).upper()}...')
    return device


def save(path, **params_to_save):
    """

    Args:
        path:
        **params_to_save:
    """
    for k, v in params_to_save.items():
        if isinstance(v, (list, tuple, dict)):
            with open(os.path.join(path, k + '.pkl'), "wb") as f:
                pickle.dump(v, f)
        elif isinstance(v, Tensor):
            torch.save(v, os.path.join(path, k + '.pt'))
        else:
            raise NotImplementedError(f'save train data not implemented for type {type(v)}')


def load(path_to_dir: str) -> dict[str: Union[Tensor, list, tuple, dict]]:
    """
    load saved train data
    Args:
        path_to_dir: path to dir with .pt->tensors, .pkl->list,tuple,dict

    Returns: dict with {param name:value} of all saved values in dict
    """
    data = {}
    for file in os.listdir(path_to_dir):
        f_split = file.split('.')
        if len(f_split) != 2:
            continue
        var_name, extension = f_split
        if extension == 'pkl':
            with open(os.path.join(path_to_dir, file), 'rb') as f:
                data[var_name] = pickle.load(f)
        elif extension == 'pt':
            data[var_name] = torch.load(os.path.join(path_to_dir, file))
        else:
            continue

    return data
