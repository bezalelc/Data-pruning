import os.path
import shutil
import sys
import time
from datetime import timedelta
from enum import Enum
from typing import Sequence, Union

import numpy as np
import torch
import torchvision
from torch import nn, optim, Tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm


class Mode(Enum):
    TRAIN = 0
    VALIDATE = 1
    TEST = 2


#
# @dataclass
# class SavedData:
#     @dataclass
#     class ProgressData:
#         @dataclass
#         class Progress:
#             loss: list[float]
#             acc: list[float]
#             scores: Tensor
#             pred: Tensor
#
#         train: Progress
#         valid: Progress
#         test: dict[str:float]
#         epochs: int
#
#     @dataclass
#     class ModelData:
#         model:nn.Module
#         optimizer,
#         scheduler
#
#


def run_epoch(model, criterion, optimizer, loader, num_classes, device, mode: Mode = Mode.TRAIN):
    model.train() if mode == Mode.TRAIN else model.eval()
    model.to(device)

    loss, loss_min, acc = .0, np.Inf, .0
    len_dataset = len(loader.dataset)
    scores = torch.empty((len(loader.dataset), num_classes))
    pred = torch.empty((len(loader.dataset),))

    for batch_idx, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        if mode == Mode.TRAIN:
            optimizer.zero_grad()

        p = model(X)
        loss_batch = criterion(p, y)
        loss += loss_batch.item()

        if mode == Mode.TRAIN:
            loss_batch.backward()
            optimizer.step()
        else:
            scores[batch_idx * loader.batch_size:(batch_idx + 1) * loader.batch_size] = p.clone().detach()

        _, pred_ = torch.max(p, 1)
        acc += torch.sum(pred_.eq(y)).item()
        pred[batch_idx * loader.batch_size:(batch_idx + 1) * loader.batch_size] = pred_

    return scores, pred, loss / len_dataset, acc / len_dataset


def train(model: nn.Module, train_loader: torch.utils.data.dataloader.DataLoader, valid_loader, test_loader, criterion,
          optimizer: torch.optim.Optimizer, scheduler, epochs: int,
          num_classes: int, device: torch.device, save_path='', verbose: bool = True):
    loss_train, loss_valid, loss_valid_min, acc_train, acc_valid = [], [], np.Inf, [], []
    scores_train, scores_valid, pred_train, pred_valid = None, None, None, None

    save_path_print = os.path.join((p1 := os.path.split(save_path))[1], p2 := os.path.split(p1[0])[1])

    for epoch in range(epochs):
        scores_train, pred_train, loss, acc = run_epoch(model, criterion, optimizer, train_loader, num_classes,
                                                        device, Mode.TRAIN)
        loss_train.append(loss), acc_train.append(acc)
        scores_valid, pred_valid, loss, acc = run_epoch(model, criterion, optimizer, valid_loader, num_classes,
                                                        device, Mode.VALIDATE)
        loss_valid.append(loss), acc_valid.append(acc)

        scheduler.step()

        # print training/validation statistics
        if verbose:
            print(f'Epoch: {epoch} Training: Loss: {loss_train[-1]:.6f} Acc: {acc_train[-1]:.6f}  '
                  f'Validation Loss: {loss_valid[-1]:.6f} Acc: {acc_valid[-1]:.6f}')

        # save model if validation loss has decreased
        if save_path and loss_valid[-1] <= loss_valid_min:
            if verbose:
                print(f'Validation loss decreased ({loss_valid_min:.6f} --> {loss_valid[-1]:.6f}).  '
                      f'Saving model to {save_path_print}')
            torch.save(model.state_dict(), save_path)
            loss_valid_min = loss_valid[-1]

    # test
    scores_test, pred_test, loss_test, acc_test = run_epoch(model, criterion, optimizer, test_loader, num_classes,
                                                            device, Mode.TEST)
    if verbose:
        print(f'Test Loss: {loss_test:.6f}')
        print(f'Accuracy: {acc_test}')

    shutil.rmtree(log_dir := os.path.join(os.path.abspath('../../../tensorboard_log'), p2), ignore_errors=True)
    tb = SummaryWriter(log_dir=log_dir)
    images, _ = next(iter(train_loader))
    tb.add_graph(model.cpu(), images)
    tb.add_images('Images', images, 0)
    for epoch in range(epochs):
        tb.add_scalars('Loss', {'train': loss_train[epoch], 'valid': loss_valid[epoch]}, epoch)
        tb.add_scalars('Accuracy', {'train': acc_train[epoch], 'valid': acc_valid[epoch]}, epoch)
    tb.close()

    return (scores_train, pred_train, loss_train, acc_train), (scores_valid, pred_valid, loss_valid, acc_valid), \
        (scores_test, pred_test, loss_test, acc_test)


# class ResNet9(nn.Module):
#
#     def __init__(self, num_classes: int) -> None:
#         super().__init__()
#         self.conv1=nn.Conv2d(,64)
#
#     def forward(self, x:):
#         return x


class ModelManager:
    DIR_ROOT = os.path.abspath(os.path.join(os.path.split(__file__)[0], '../../../   '))
    DIR_ROOT_SAVE = os.path.join(DIR_ROOT, 'models_data')
    DIR_ROOT_LOG = os.path.join(DIR_ROOT, 'tensorboard_log')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, num_classes: int, model_name: str = 'tmp', dir_: str = '', load: bool = False) -> None:
        super().__init__()

        self.path_saved_data = os.path.join(self.DIR_ROOT_SAVE, dir_, model_name)
        self.path_saved_model = os.path.join(self.path_saved_data, 'model.pt')
        self.path_saved_progress = os.path.join(self.path_saved_data, 'progress.pt')
        self.path_saved_other = os.path.join(self.path_saved_data, 'other.pt')
        self.path_log = os.path.join(self.DIR_ROOT_LOG, dir_, model_name)
        self.name = model_name
        self.dir = dir_
        self.num_classes: int = num_classes

        self.model: torchvision.models.resnet.ResNet = models.resnet18(weights=None).to(self.DEVICE)
        self.model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                           bias=False).to(self.DEVICE)
        self.model.maxpool = nn.Identity()
        # self.model.fc = nn.Linear(self.model.fc.in_features, num_classes).to(self.DEVICE)
        self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, 256),
                                      nn.Linear(256, num_classes)).to(self.DEVICE)
        self.criterion: torch.nn.modules.loss.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.optimizer: torch.optim.Adam = optim.Adam(self.model.parameters(), lr=5e-3)  # lr=1e-4
        self.scheduler: torch.optim.lr_scheduler.StepLR = \
            optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)

        if load:
            self.load_model()

            self.data: dict = torch.load(self.path_saved_progress)
            self.data_other: dict = torch.load(self.path_saved_other)
        else:
            shutil.rmtree(self.path_log, ignore_errors=True)
            shutil.rmtree(self.path_saved_data, ignore_errors=True)
            os.makedirs(self.path_saved_data)

            self.data: dict = {
                'train': {'loss': [], 'acc': [], 'scores': None, 'pred': None},
                'valid': {'loss': [], 'acc': [], 'scores': None, 'pred': None},
                'test': {'loss': np.Inf, 'acc': 0, 'scores': None, 'pred': None},
                'epochs': 0,
            }
            self.data_other: dict = {}

        self.epochs: int = self.data['epochs']
        self.tb = SummaryWriter(log_dir=self.path_log)
        self.model.eval()

    def run_epoch(self, loader, mode: Mode = Mode.TRAIN):
        model, optimizer, criterion, scheduler = self.model, self.optimizer, self.criterion, self.scheduler
        num_classes, device = self.num_classes, self.DEVICE
        model.train() if mode == Mode.TRAIN else model.eval()

        loss, loss_min, acc = .0, np.Inf, .0
        len_dataset = len(loader.dataset)
        scores = torch.empty((len(loader.dataset), num_classes))
        pred = torch.empty((len(loader.dataset),))

        mode_str = {Mode.TRAIN: 'Train', Mode.VALIDATE: 'Validate', Mode.TEST: 'Test'}
        progress = tqdm(unit=' batch', file=sys.stdout, total=len_dataset // loader.batch_size, position=0, leave=False,
                        bar_format=f"Epoch {self.epochs} {mode_str[mode]}: {{bar:70}}  {{n_fmt}}/{{total_fmt}} "
                                   f"[{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]")

        for batch_idx, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)

            if mode == Mode.TRAIN:
                optimizer.zero_grad()

            p = model(X)
            loss_batch = criterion(p, y)
            loss += loss_batch.item() / len_dataset

            if mode == Mode.TRAIN:
                loss_batch.backward()
                optimizer.step()
            else:
                p = scores[batch_idx * loader.batch_size:(batch_idx + 1) * loader.batch_size] = p.clone().detach()

            _, pred_ = torch.max(p, 1)
            acc += torch.sum(pred_.eq(y)).item() / len_dataset
            pred[batch_idx * loader.batch_size:(batch_idx + 1) * loader.batch_size] = pred_

            progress.update(1)
            progress.set_postfix(loss=f'{loss:.2}', acc=f'{int(100. * acc)}%')

        progress.close()
        if mode == Mode.TRAIN:
            scheduler.step()

        return scores, pred, loss, acc

    def train(self, train_loader, valid_loader, test_loader, epochs: int, verbose: bool = True):
        loss_train, loss_valid, acc_train, acc_valid = [], [], [], []
        loss_valid_min = np.min(self.data['valid']['loss']) if self.data['valid']['loss'] else np.Inf
        did_not_improve_counter = 0

        scores_train, scores_valid, pred_train, pred_valid = None, None, None, None

        for epoch in range(self.epochs, self.epochs + epochs):
            scores_train, pred_train, loss, acc = self.run_epoch(train_loader, Mode.TRAIN)
            loss_train.append(loss), acc_train.append(acc)
            scores_valid, pred_valid, loss, acc = self.run_epoch(valid_loader, Mode.VALIDATE)
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
                    self.load_model()
                    did_not_improve_counter = 0
                else:
                    did_not_improve_counter += 1

            self.epochs += 1

        # test
        scores_test, pred_test, loss_test, acc_test = self.run_epoch(test_loader, Mode.TEST)
        if verbose:
            print(f'Test Loss: {loss_test:.6f}')
            print(f'Accuracy: {acc_test}')

        self._save_progress(train_loader, scores_train, pred_train, loss_train, acc_train, scores_valid, pred_valid,
                            loss_valid, acc_valid, scores_test, pred_test, loss_test, acc_test)

    def _save_progress(self,
                       train_loader, scores_train, pred_train, loss_train, acc_train,
                       scores_valid, pred_valid, loss_valid, acc_valid,
                       scores_test, pred_test, loss_test, acc_test
                       ):
        model, tb, prev_epochs = self.model, self.tb, self.data['epochs']

        if not prev_epochs:
            images, _ = next(iter(train_loader))
            tb.add_graph(model.cpu(), images.cpu() if isinstance(images, Tensor) else images)
            tb.add_images('Images', images, 0)
        for epoch in range(len(loss_train)):
            tb.add_scalars('Loss', {'train': loss_train[epoch], 'valid': loss_valid[epoch]}, epoch + prev_epochs)
            tb.add_scalars('Accuracy', {'train': acc_train[epoch], 'valid': acc_valid[epoch]}, epoch + prev_epochs)
        tb.close()

        self.data['train']['loss'] += loss_train
        self.data['train']['acc'] += acc_train
        self.data['train']['scores'] = scores_train
        self.data['train']['pred'] = pred_train
        self.data['valid']['loss'] += loss_valid
        self.data['valid']['acc'] += acc_valid
        self.data['valid']['scores'] = scores_valid
        self.data['valid']['pred'] = pred_valid
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
        path = lambda f: os.path.join(ModelManager.DIR_ROOT, 'models_data', f, 'progress.pt')

        data_models = {model.name: torch.load(path(model.name)) for model in models_}
        epochs = len(np.min([model.data['epochs'] for model in models_]))

        tb = SummaryWriter(log_dir=os.path.join(ModelManager.DIR_ROOT_LOG, dir_log_name))
        for i in range(epochs):
            tb.add_scalars('models loss', {**{f'{k} train': v['train']['loss'][i] for k, v in data_models.items()},
                                           **{f'{k} valid': v['valid']['loss'][i] for k, v in data_models.items()}}, i)
            tb.add_scalars('models acc', {**{f'{k} train': v['train']['acc'][i] for k, v in data_models.items()},
                                          **{f'{k} valid': v['valid']['acc'][i] for k, v in data_models.items()}}, i)
        tb.close()

    @staticmethod
    def ensemble_predict(ensemble,loader):
        ensemble_scores=[]

def main():
    from code.src.utils.common import get_loader
    from code.src.utils.dataset import GPUDataset, get_cifar
    # from ..config import *

    # NUM_CLASSES = 10
    BATCH_SIZE = 25
    NUM_TRAIN = 50000
    NUM_TEST = 10000
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

    dataset = GPUDataset(load=True)
    loader_train = get_loader(dataset, np.arange(NUM_TRAIN), BATCH_SIZE)
    loader_test = get_loader(dataset, np.arange(NUM_TRAIN, NUM_TRAIN + NUM_TEST), BATCH_SIZE)
    dataset.to('cuda')
    # dataset_train, dataset_test = get_cifar(GPUDataset.PATH_DATASETS, cifar100=True)
    # print([n / 255. for n in [129.3, 124.1, 112.4]], [n / 255. for n in [68.2, 65.4, 70.4]])

    # Stick all the images together to form a 1600000 X 32 X 3 array
    # x = np.concatenate([np.asarray(dataset_train[i][0]) for i in range(len(dataset_train))] +
    #                    [np.asarray(dataset_test[i][0]) for i in range(len(dataset_test))])

    # the the mean and std
    # loader_train = get_loader(dataset_train, np.arange(NUM_TRAIN), BATCH_SIZE)
    # loader_test = get_loader(dataset_test, np.arange(NUM_TEST), BATCH_SIZE)
    # print(len(dataset_train.classes))
    model_manager = ModelManager(100)
    # print(model_manager.model)

    # start_time = time.monotonic()
    model_manager.train(loader_train, loader_test, loader_test, EPOCHS)
    # end_time = time.monotonic()
    # print(timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    main()
