import os

import torch
from torchvision import models
from torch import nn, optim


def get_loader(dataset, idx,batch_size, shuffle=True):
    subset = torch.utils.data.Subset(dataset, idx)
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def get_model_resnet18_cifar10():
    model = models.resnet18(weights=None)  # ,pretrained=False
    model.fc = nn.Linear(model.fc.in_features, 10)
    # model.to(DEVICE)
    # model.load_state_dict(torch.load('ResNet18.pt'))

    # lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.0001)

    return model, criterion, optimizer


def create_saved_data_dir(file):
    dir_, f = os.path.split(file)
    return os.path.abspath(os.path.join(dir_, '../../../', 'models_data', f.split('.')[0]))

def get_device():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if str(device) == 'cuda':
        print('CUDA is available!  Training on  GPU...')
    else:
        print(f'CUDA is not available.  Training on {str(device).upper()}...')
    return device
