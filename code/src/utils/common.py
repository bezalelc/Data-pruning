import os
from typing import Callable

import torch
from torch import nn, optim
from torchvision import models


def get_loader(dataset, idx, batch_size: int, shuffle: bool = True) -> torch.utils.data.DataLoader:
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
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def get_model_resnet18_cifar10() -> tuple:
    """

    Returns:
    """
    model = models.resnet18(weights=None)  # ,pretrained=False
    model.fc = nn.Linear(model.fc.in_features, 10)
    # model.to(DEVICE)
    # model.load_state_dict(torch.load('ResNet18.pt'))

    # lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.0001)

    return model, criterion, optimizer


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
    return lambda f: os.path.join(path_to_save, f)


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
