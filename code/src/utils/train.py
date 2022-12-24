import os.path
import shutil
import sys
from enum import Enum
from typing import Sequence

import numpy as np
import torch
import torchvision
from torch import nn, optim, Tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm
from torch.nn import Module, Conv2d, Linear, BatchNorm2d, AvgPool2d, ReLU, Sequential, AdaptiveAvgPool2d


class Mode(Enum):
    TRAIN = 0
    VALIDATE = 1
    TEST = 2


DIR_ROOT = os.path.abspath(os.path.join(os.path.split(__file__)[0], '../../../   '))
DIR_ROOT_SAVE = os.path.join(DIR_ROOT, 'models_data')
DIR_ROOT_LOG = os.path.join(DIR_ROOT, 'tensorboard_log')

"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
    
    xxxxxxxxxxxxxxx
"""


class BasicBlock(Module):
    def __init__(self, in_, out, kernel_size=(3, 3), stride=(1, 1)):
        super().__init__()
        self.conv1 = Conv2d(in_, out, kernel_size=kernel_size, stride=stride, padding=(1, 1), bias=False)
        self.bn1 = BatchNorm2d(out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv2d(out, out, kernel_size=kernel_size, stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = BatchNorm2d(out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x


class ResNet18(Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = ReLU(inplace=True)
        #   (maxpool): Identity()
        stride = (2, 2)
        # with    stride=1                          stride=2
        # layer 1 torch.Size([25, 128, 32, 32])     torch.Size([25, 64, 32, 32])
        # layer 2 torch.Size([25, 128, 32, 32])     torch.Size([25, 128, 16, 16])
        # layer 3 torch.Size([25, 256, 32, 32])     torch.Size([25, 256, 8, 8])
        # layer 4 torch.Size([25, 512, 32, 32])     torch.Size([25, 512, 4, 4])
        # avd     torch.Size([25, 512, 1, 1])       torch.Size([25, 512, 1, 1])
        # view    torch.Size([25, 512])             torch.Size([25, 512])
        self.layer1 = Sequential(BasicBlock(in_=64, out=64, kernel_size=(3, 3), stride=(1, 1)),
                                 BasicBlock(in_=64, out=64, kernel_size=(3, 3), stride=(1, 1)))
        # => torch.Size([25, 64, 32, 32])
        self.layer2 = Sequential(BasicBlock(in_=64, out=128, kernel_size=(3, 3), stride=(1, 1)),
                                 BasicBlock(in_=128, out=128, kernel_size=(3, 3), stride=(1, 1)))
        # torch.Size([25, 128, 16, 16])
        self.layer3 = Sequential(BasicBlock(in_=128, out=256, kernel_size=(3, 3), stride=(1, 1)),
                                 BasicBlock(in_=256, out=256, kernel_size=(3, 3), stride=(1, 1)))
        # => torch.Size([25, 256, 8, 8])
        self.layer4 = Sequential(BasicBlock(in_=256, out=512, kernel_size=(3, 3), stride=stride),
                                 BasicBlock(in_=512, out=512, kernel_size=(3, 3), stride=(1, 1)))
        # => torch.Size([25, 512, 4, 4])
        self.avgpool = AdaptiveAvgPool2d(output_size=(1, 1))
        # => torch.Size([25, 512, 1, 1])
        # after reshape
        # => torch.Size([25, 512])
        self.fc = Linear(in_features=512, out_features=num_classes, bias=True)

    #   (layer2): Sequential(
    #     (0): BasicBlock(
    #       (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (downsample): Sequential(
    #         (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
    #         (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       )
    #     )
    #     (1): BasicBlock(
    #       (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     )
    #   )
    #   (layer3): Sequential(
    #     (0): BasicBlock(
    #       (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (downsample): Sequential(
    #         (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
    #         (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       )
    #     )
    #     (1): BasicBlock(
    #       (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     )
    #   )
    #   (layer4): Sequential(
    #     (0): BasicBlock(
    #       (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (downsample): Sequential(
    #         (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
    #         (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       )
    #     )
    #     (1): BasicBlock(
    #       (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     )
    #   )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

        # 1
        # torch.Size([25, 64, 32, 32])
        # 2
        # torch.Size([25, 128, 16, 16])
        # 3
        # torch.Size([25, 128, 16, 16])
        # 4
        # torch.Size([25, 256, 8, 8])
        # 5
        # torch.Size([25, 512, 4, 4])
        # 6
        # torch.Size([25, 512, 4, 4])
        # 7
        # torch.Size([25, 100])


class ModelManager:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, num_classes: int, model_name: str = 'tmp', dir_: str = '', load: bool = True) -> None:
        super().__init__()

        self.path_saved_data = os.path.join(DIR_ROOT_SAVE, dir_, model_name)
        self.path_saved_model = os.path.join(self.path_saved_data, 'model.pt')
        self.path_saved_progress = os.path.join(self.path_saved_data, 'progress.pt')
        self.path_saved_other = os.path.join(self.path_saved_data, 'other.pt')
        self.path_log = os.path.join(DIR_ROOT_LOG, dir_, model_name)
        self.name = model_name
        self.dir = dir_
        self.num_classes: int = num_classes

        # self.model: Module = ResNet9(100)
        self.model: Module = ResNet18(100)
        # self.model: torchvision.models.resnet.ResNet = models.resnet18(weights=None)
        # self.model: torchvision.models.resnet.ResNet = models.resnet50(weights=None).to(self.DEVICE)
        # self.model.conv1 = Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # self.model.maxpool = nn.Identity()
        # self.model.layer2[0].conv1 = Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # self.model.layer3[0].conv1 = Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # self.model.layer2[0].downsample = nn.Identity()
        # self.model.layer3[0].downsample = nn.Identity()
        # self.model.layer4[0].downsample = nn.Identity()
        # self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.fc = nn.Sequential(Linear(in_features=512, out_features=200, bias=True),
                                      nn.Linear(200, num_classes, bias=True))  # .to(self.DEVICE)
        self.criterion: torch.nn.modules.loss.CrossEntropyLoss = nn.CrossEntropyLoss()  # .to(self.DEVICE)
        self.model.to(self.DEVICE)
        self.optimizer: torch.optim.SGD = optim.SGD(self.model.parameters(), lr=1e-3, momentum=.9)  # lr=1e-1
        # self.optimizer: torch.optim.Adam = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=.5e-4)  # lr=1e-4
        self.scheduler: torch.optim.lr_scheduler.MultiStepLR = \
            optim.lr_scheduler.MultiStepLR(self.optimizer,
                                           milestones=[60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115], gamma=0.3)
        # optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

        if load:
            self.load_model()

            self.data: dict = torch.load(self.path_saved_progress)
            self.data_other: dict = torch.load(self.path_saved_other)
        else:
            shutil.rmtree(self.path_log, ignore_errors=True)
            shutil.rmtree(self.path_saved_data, ignore_errors=True)
            os.makedirs(self.path_saved_data)

            self.data: dict = {
                'train': {'loss': [], 'acc': []},
                'valid': {'loss': [], 'acc': []},
                'test': {'loss': np.Inf, 'acc': 0, 'scores': None, 'pred': None},
                'epochs': 0,
            }
            self.data_other: dict = {}

        self.epochs: int = self.data['epochs']
        self.tb = SummaryWriter(log_dir=self.path_log)
        self.model.eval(), self.model.cpu()

    def run_epoch(self, loader, mode: Mode = Mode.TRAIN):
        model, optimizer, criterion, scheduler = self.model, self.optimizer, self.criterion, self.scheduler
        num_classes, device = self.num_classes, self.DEVICE
        model.train() if mode == Mode.TRAIN else model.eval()

        loss, loss_min, acc = .0, np.Inf, .0
        len_dataset = len(loader.dataset)

        if mode == Mode.TEST:
            scores = torch.empty((len(loader.dataset), num_classes), device=self.DEVICE)
            pred = torch.empty((len(loader.dataset),), device=self.DEVICE)
        else:
            scores, pred = None, None

        mode_str = {Mode.TRAIN: 'Train', Mode.VALIDATE: 'Validate', Mode.TEST: 'Test'}
        progress = tqdm(unit=' batch', file=sys.stdout, total=len_dataset // loader.batch_size, position=0, leave=False,
                        bar_format=f"Epoch {self.epochs} {mode_str[mode]}: {{bar:60}}  {{n_fmt}}/{{total_fmt}} "
                                   f"[{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]")

        for batch_idx, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)

            if mode == Mode.TRAIN:
                optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                p = model(X)
                loss_batch = criterion(p, y)

            loss += loss_batch.item() / len_dataset

            if mode == Mode.TRAIN:
                loss_batch.backward()
                optimizer.step()

            if mode == Mode.TEST:
                scores[batch_idx * loader.batch_size:(batch_idx + 1) * loader.batch_size] = p.clone().detach()

            _, pred_ = torch.max(p, 1)
            acc += torch.sum(pred_.eq(y)).item() / len_dataset
            if mode == Mode.TEST:
                pred[batch_idx * loader.batch_size:(batch_idx + 1) * loader.batch_size] = pred_

            progress.update(1)
            progress.set_postfix(loss=f'{loss:.2}', acc=f'{acc:.2%}')

        progress.close()
        if mode == Mode.TRAIN:
            scheduler.step()

        # print(pred.device, )
        return scores, pred, loss, acc

    def train(self, train_loader, valid_loader, test_loader, epochs: int, verbose: bool = True):
        loss_train, loss_valid, acc_train, acc_valid = [], [], [], []
        loss_valid_min = np.min(self.data['valid']['loss']) if self.data['valid']['loss'] else np.Inf
        did_not_improve_counter = 0

        # general configurations
        self.model.to(self.DEVICE)
        benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = True

        # with torch.autograd.profiler.profile(enabled=False), torch.autograd.detect_anomaly(check_nan=False):
        for epoch in range(self.epochs, self.epochs + epochs):
            _, _, loss, acc = self.run_epoch(train_loader, Mode.TRAIN)
            loss_train.append(loss), acc_train.append(acc)
            _, _, loss, acc = self.run_epoch(valid_loader, Mode.VALIDATE)
            loss_valid.append(loss), acc_valid.append(acc)

            if verbose:
                print(f'Epoch: {epoch} Training: Loss: {loss_train[-1]:.6f} Acc: {acc_train[-1]:.6f}  '
                      f'Validation Loss: {loss_valid[-1]:.6f} Acc: {acc_valid[-1]:.6f}')

            # save model if validation loss has decreased
            if loss_valid[-1] <= loss_valid_min:
                if verbose:
                    print(f'Validation loss decreased ({loss_valid_min:.6f} --> {loss_valid[-1]:.6f}).  '
                          f'Saving model to models_data/{os.path.join(self.dir, self.name)}')
                    self.save_model()
                loss_valid_min, did_not_improve_counter = loss_valid[-1], 0
            else:
                if did_not_improve_counter == 5:
                    print(f'Load model: did_not_improve_counter={did_not_improve_counter}')
                    # self.load_model()
                    did_not_improve_counter = 0
                else:
                    did_not_improve_counter += 1

            self.epochs += 1

        # test
        scores_test, pred_test, loss_test, acc_test = self.run_epoch(test_loader, Mode.TEST)
        if verbose:
            print(f'Test Loss: {loss_test:.6f}')
            print(f'Accuracy: {acc_test}')

        # restore
        self.model.cpu()
        torch.backends.cudnn.benchmark = benchmark

        self._save_progress(loss_train, acc_train, loss_valid, acc_valid, scores_test, pred_test, loss_test, acc_test)

    def _save_progress(self, loss_train, acc_train, loss_valid, acc_valid,
                       scores_test, pred_test, loss_test, acc_test
                       ):
        model, tb, prev_epochs = self.model, self.tb, self.data['epochs']

        # if not prev_epochs:
        # images, _ = next(iter(train_loader))
        # tb.add_graph(model, images.cpu() if isinstance(images, Tensor) else images)
        # tb.add_images('Images', images, 0)
        for epoch in range(len(loss_train)):
            tb.add_scalars('Loss', {'train': loss_train[epoch], 'valid': loss_valid[epoch]}, epoch + prev_epochs)
            tb.add_scalars('Accuracy', {'train': acc_train[epoch], 'valid': acc_valid[epoch]}, epoch + prev_epochs)
        tb.close()

        self.data['train']['loss'] += loss_train
        self.data['train']['acc'] += acc_train
        # self.data['train']['scores'] = scores_train
        # self.data['train']['pred'] = pred_train
        self.data['valid']['loss'] += loss_valid
        self.data['valid']['acc'] += acc_valid
        # self.data['valid']['scores'] = scores_valid
        # self.data['valid']['pred'] = pred_valid
        self.data['test']['loss'] = loss_test
        self.data['test']['acc'] = acc_test
        self.data['valid']['scores'] = scores_test
        self.data['valid']['pred'] = pred_test
        self.data['epochs'] = self.epochs

        # data = {
        #     'train':
        #         {'loss': self.data['train']['loss'] + loss_train, 'acc': self.data['train']['acc'] + acc_train,
        #          'scores': scores_train, 'pred': pred_train},
        #     'valid':
        #         {'loss': self.data['valid']['loss'] + loss_valid, 'acc': self.data['valid']['acc'] + acc_valid,
        #          'scores': scores_valid, 'pred': pred_valid},
        #     'test':
        #         {'loss': loss_test, 'acc': acc_test, 'scores': scores_test, 'pred': pred_test},
        #     'epochs': self.epochs,
        # }
        self.save()

    def save_other(self, dict_: dict):
        self.data_other = {**self.data_other, **dict_}
        torch.save(self.data_other, self.path_saved_other)

    def save_model(self):
        torch.save({'model': self.model.state_dict(), 'scheduler': self.scheduler.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, self.path_saved_model)

    def save(self, data: dict = None, data_other: dict = None):
        torch.save({**self.data, **(data if data else {})}, self.path_saved_progress)
        torch.save({**self.data_other, **(data_other if data_other else {})}, self.path_saved_other)

    def load_model(self):
        model_data = torch.load(self.path_saved_model, map_location=self.DEVICE)
        self.model.load_state_dict(model_data['model'])
        self.optimizer.load_state_dict(model_data['optimizer'])
        self.scheduler.load_state_dict(model_data['scheduler'])

    @staticmethod
    def save_models_log(models_: Sequence, dir_log_name: str = 'models_compare'):
        """
        Compare loss and accuracy for multiple models in tensorboard

        Args:
            models_: Sequence[ModelManager]
            dir_log_name: log will save in path Data-pruning/tensorboard_log/{dir_log_name}
                for activate tensorboard run tensorboard --logdir=tensorboard_log/{dir_log_name} from
                Data-pruning dir
        """
        path = lambda f: os.path.join(DIR_ROOT, 'models_data', dir_log_name, f, 'progress.pt')

        data_models = {model.name: torch.load(path(model.name)) for model in models_}
        epochs = np.min([model.data['epochs'] for model in models_])

        tb = SummaryWriter(log_dir=os.path.join(DIR_ROOT_LOG, dir_log_name))
        for i in range(epochs):
            tb.add_scalars('models loss', {**{f'{k} train': v['train']['loss'][i] for k, v in data_models.items()},
                                           **{f'{k} valid': v['valid']['loss'][i] for k, v in data_models.items()}}, i)
            tb.add_scalars('models acc', {**{f'{k} train': v['train']['acc'][i] for k, v in data_models.items()},
                                          **{f'{k} valid': v['valid']['acc'][i] for k, v in data_models.items()}}, i)
        tb.close()

    @staticmethod
    def ensemble_predict(ensemble: Sequence, loader):
        ensemble_scores = [model.run_epoch(loader, Mode.TEST)[0] for model in ensemble]


class EnsembleManager:
    def __init__(self, num_classes, dir_: str = 'ensemble', load: bool = True):
        self.path_log = os.path.join(DIR_ROOT_LOG, dir_)
        self.path_saved_data = os.path.join(DIR_ROOT_SAVE, dir_, 'data.pt')

        if load:
            pass
        else:
            shutil.rmtree(self.path_log, ignore_errors=True)
            shutil.rmtree(self.path_saved_data, ignore_errors=True)

            self.data: dict[str:Tensor] = {
                'ensemble_softmax': None,
                'ensemble_pred': None,
                'ensemble_pred_sum': None,
                'ensemble_var': None,
                'el2n_scores': None
            }

            self.ensemble: list[ModelManager] = \
                [ModelManager(num_classes, f'model_{i}', dir_, load) for i in range(10)]
            self.epochs: int = 0

            self.tb: SummaryWriter = SummaryWriter(log_dir=self.path_log)

    def train(self, train_loader, valid_loader, test_loader, loader_train_ordered, Y_train, epochs: int,
              verbose: bool = True):

        num_train, ensemble_len = 50000, 10
        ensemble_softmax = torch.empty((ensemble_len, num_train, self.ensemble[0].num_classes))
        ensemble_pred = torch.empty((num_train, ensemble_len), dtype=torch.bool)

        for i, model in enumerate(self.ensemble):
            print(f'------------   model {i}   -------------------')
            model.train(train_loader, valid_loader, test_loader, epochs)
            scores, pred, loss, acc = model.run_epoch(loader_train_ordered, mode=Mode.TEST)
            ensemble_softmax[i] = nn.functional.softmax(scores.clone().detach().cpu(), dim=1)
            ensemble_pred[:, i] = torch.Tensor(pred.type(torch.int8) == Y_train).clone().detach().cpu()
        self.epochs += epochs

        ensemble_pred_sum = torch.sum(ensemble_pred, dim=1)
        ensemble_var = ensemble_softmax.var(dim=0)
        # el2n_scores = get_el2n_scores(Y_train, ensemble_softmax)

        self.data: dict[str:Tensor] = {
            'ensemble_softmax': ensemble_softmax,
            'ensemble_pred': ensemble_pred,
            'ensemble_pred_sum': ensemble_pred_sum,
            'ensemble_var': ensemble_var,
            'el2n_scores': None
        }
        torch.save(self.data, self.path_saved_data)

        return ensemble_softmax, ensemble_pred  # , ensemble_pred_sum, ensemble_var, el2n_scores

    # def save_models_log(self):
    #     """
    #     Compare loss and accuracy for multiple models in tensorboard
    #
    #     Args:
    #         models_: Sequence[ModelManager]
    #         dir_log_name: log will save in path Data-pruning/tensorboard_log/{dir_log_name}
    #             for activate tensorboard run tensorboard --logdir=tensorboard_log/{dir_log_name} from
    #             Data-pruning dir
    #     """
    #     path = lambda f: os.path.join(DIR_ROOT, 'models_data', f, 'progress.pt')
    #
    #     data_models = {model.name: torch.load(path(model.name)) for model in self.ensemble}
    #     epochs = len(np.min([model.data['epochs'] for model in models_]))
    #
    #     # tb = SummaryWriter(log_dir=self.path_log)
    #     for i in range(epochs):
    #         self.tb.add_scalars('models loss', {**{f'{k} train': v['train']['loss'][i] for k, v in data_models.items()},
    #                                             **{f'{k} valid': v['valid']['loss'][i] for k, v in
    #                                                data_models.items()}}, i)
    #         self.tb.add_scalars('models acc', {**{f'{k} train': v['train']['acc'][i] for k, v in data_models.items()},
    #                                            **{f'{k} valid': v['valid']['acc'][i] for k, v in data_models.items()}},
    #                             i)
    #     self.tb.close()

    # def ensemble_predict( loader):
    #     ensemble_scores = [model.run_epoch(loader, Mode.TEST)[0] for model in ensemble]


def main():
    from code.src.utils.common import get_loader
    from code.src.utils.dataset import GPUDataset
    # from ..config import *

    # NUM_CLASSES = 10
    BATCH_SIZE = 5
    NUM_TRAIN = 50
    NUM_TEST = 10
    EPOCHS = 10
    # reg: worker 8,pin_mem (00:46,00:10)
    # reg: worker 2,pin_mem (00:41,00:06)
    # reg: worker 1,pin_mem (00:38,00:04)
    # reg: worker 8         (00:56,00:17)
    # reg: worker 1         (00:39,00:04)
    # reg: pin_mem          (00:50,00:04)
    # reg:                  (00:45,00:04)
    # GPUDataset            (00:36,00:02)
    # GPUDataset .to()      (00:36,00:02)
    # 100                   (00:13,00:00)
    # 1000                  (00:10,00:00)

    # dataset = GPUDataset(load=True, cifar100=True)
    # loader_train = get_loader(dataset, np.arange(NUM_TRAIN), BATCH_SIZE)
    # loader_train_ordered = get_loader(dataset, np.arange(NUM_TRAIN), BATCH_SIZE, False)
    # loader_test = get_loader(dataset, np.arange(NUM_TRAIN, NUM_TRAIN + NUM_TEST), BATCH_SIZE)
    # dataset.to('cuda')

    # epochs_pred = torch.empty((EPOCHS, NUM_TRAIN), dtype=torch.int8)
    # # change_counter = torch.zeros(NUM_TRAIN, dtype=torch.int8)
    # # print(ResNet(BasicBlock, [3, 3, 3], 100))
    # model_manager = ModelManager(100, load=False)
    #
    # # print(model_manager.model)
    # # model_manager.train(loader_train, loader_test, loader_test, 1)
    #


if __name__ == '__main__':
    main()
