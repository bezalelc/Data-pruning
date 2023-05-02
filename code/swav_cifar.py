import os
import time
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from lightly.data import (
    LightlyDataset,
    SwaVCollateFunction,
    collate,
)
from lightly.loss import SwaVLoss
from lightly.models import ResNetGenerator
from lightly.models.modules import heads
from lightly.utils.benchmarking import BenchmarkModule
from pytorch_lightning.loggers import TensorBoardLogger

from train import DIR_ROOT_SAVE
from utils import PATH_DATASETS

warnings.filterwarnings("ignore", category=UserWarning)

logs_root_dir = os.path.join(os.getcwd(), "benchmark_logs")

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 200
num_workers = 4
knn_k = 20 # 200
knn_t = 0.1
FEATURES = 512  # 512
NUM_CLASSES = 100
NUM_TRAIN = 50000
classes = NUM_CLASSES
path_dataset = os.path.join(PATH_DATASETS, f'cifar{NUM_CLASSES}images')
path_to_train = os.path.join(path_dataset, 'train')
path_to_test = os.path.join(path_dataset, 'test')

save_to = os.path.join(DIR_ROOT_SAVE, f'swav_cifar{NUM_CLASSES}_{FEATURES}')
save_to_model = os.path.join(save_to, 'model.pt')
if not os.path.exists(save_to):
    # shutil.rmtree(save_to)
    os.makedirs(save_to)

# Set to True to enable Distributed Data Parallel training.
distributed = False

# Set to True to enable Synchronized Batch Norm (requires distributed=True).
# If enabled the batch norm is calculated over all gpus, otherwise the batch
# norm is only calculated from samples on the same gpu.
sync_batchnorm = False

# Set to True to gather features from all gpus before calculating
# the loss (requires distributed=True).
# If enabled then the loss on every gpu is calculated with features from all
# gpus, otherwise only features from the same gpu are used.
gather_distributed = False

# benchmark
n_runs = 1  # optional, increase to create multiple runs and report mean + std
batch_size = 512
lr_factor = batch_size / 512  # scales the learning rate linearly with batch size

# use a GPU if available
gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

if distributed:
    distributed_backend = "ddp"
    # reduce batch size for distributed training
    batch_size = batch_size // gpus
else:
    distributed_backend = None
    # limit to single gpu if not using distributed training
    gpus = min(gpus, 1)

# Multi crop augmentation for SwAV, additionally, disable blur for cifar10
swav_collate_fn = SwaVCollateFunction(
    crop_sizes=[32],
    crop_counts=[2],  # 2 crops @ 32x32px
    crop_min_scales=[0.14],
    gaussian_blur=0,
)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=collate.imagenet_normalize["mean"],
            std=collate.imagenet_normalize["std"],
        ),
    ]
)

dataset_train_ssl = LightlyDataset(input_dir=path_to_train)
# we use test transformations for getting the feature for kNN on train data
dataset_train_kNN = LightlyDataset(input_dir=path_to_train, transform=test_transforms)
dataset_test = LightlyDataset(input_dir=path_to_test, transform=test_transforms)


def get_data_loaders(batch_size: int):
    """Helper method to create dataloaders for ssl, kNN train and kNN test

    Args:
        batch_size: Desired batch size for all dataloaders
    """
    col_fn = swav_collate_fn
    dataloader_train_ssl = torch.utils.data.DataLoader(
        dataset_train_ssl,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=col_fn,
        drop_last=True,
        num_workers=num_workers,
    )

    dataloader_train_kNN = torch.utils.data.DataLoader(
        dataset_train_kNN,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return dataloader_train_ssl, dataloader_train_kNN, dataloader_test


class SwaVModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator("resnet-18")
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )

        self.projection_head = heads.SwaVProjectionHead(512, 512, 128)
        self.prototypes = heads.SwaVPrototypes(128, FEATURES)  # use 512 prototypes

        self.criterion = SwaVLoss(sinkhorn_gather_distributed=gather_distributed)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        return self.prototypes(x)

    def training_step(self, batch, batch_idx):
        # normalize the prototypes so they are on the unit sphere
        self.prototypes.normalize()

        # the multi-crop dataloader returns a list of image crops where the
        # first two items are the high resolution crops and the rest are low
        # resolution crops
        multi_crops, _, _ = batch
        multi_crop_features = [self.forward(x) for x in multi_crops]

        # split list of crop features into high and low resolution
        high_resolution_features = multi_crop_features[:2]
        low_resolution_features = multi_crop_features[2:]

        # calculate the SwaV loss
        loss = self.criterion(high_resolution_features, low_resolution_features)

        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(),
            lr=1e-3 * lr_factor,
            weight_decay=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


bench_results = dict()

if __name__ == '__main__':
    experiment_version = None

    runs = []
    model_name = SwaVModel.__name__.replace("Model", "")
    seed = 0

    pl.seed_everything(seed)
    dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(batch_size=batch_size)
    swav = SwaVModel(dataloader_train_kNN, classes)

    # Save logs to: {CWD}/benchmark_logs/cifar10/{experiment_version}/{model_name}/
    # If multiple runs are specified a subdirectory for each run is created.
    sub_dir = model_name if n_runs <= 1 else f"{model_name}/run{seed}"
    logger = TensorBoardLogger(save_dir=os.path.join(logs_root_dir, f"cifar{NUM_CLASSES}"), name="", sub_dir=sub_dir,
                               version=experiment_version)
    if experiment_version is None:
        # Save results of all models under same version directory
        experiment_version = logger.version
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints")
    )
    trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus, default_root_dir=logs_root_dir, strategy=distributed_backend,
                         sync_batchnorm=sync_batchnorm, logger=logger, callbacks=[checkpoint_callback])
    start = time.time()
    trainer.fit(
        swav,
        train_dataloaders=dataloader_train_ssl,
        val_dataloaders=dataloader_test,
    )
    end = time.time()
    torch.save({'model': swav.state_dict()}, save_to_model)
    run = {
        "model": model_name,
        "batch_size": batch_size,
        "epochs": max_epochs,
        "max_accuracy": swav.max_accuracy,
        "runtime": end - start,
        "gpu_memory_usage": torch.cuda.max_memory_allocated(),
        "seed": seed,
    }
    runs.append(run)
    print(run)

    # delete model and trainer + free up cuda memory
    # del benchmark_model
    # del trainer
    # torch.cuda.reset_peak_memory_stats()
    # torch.cuda.empty_cache()
    print(4, os.path.exists(save_to_model))
    bench_results[model_name] = runs

    # print results table
    header = (
        f"| {'Model':<13} | {'Batch Size':>10} | {'Epochs':>6} "
        f"| {'KNN Test Accuracy':>18} | {'Time':>10} | {'Peak GPU Usage':>14} |"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for model, results in bench_results.items():
        runtime = np.array([result["runtime"] for result in results])
        runtime = runtime.mean() / 60  # convert to min
        accuracy = np.array([result["max_accuracy"] for result in results])
        gpu_memory_usage = np.array([result["gpu_memory_usage"] for result in results])
        gpu_memory_usage = gpu_memory_usage.max() / (1024 ** 3)  # convert to gbyte

        if len(accuracy) > 1:
            accuracy_msg = f"{accuracy.mean():>8.3f} +- {accuracy.std():>4.3f}"
        else:
            accuracy_msg = f"{accuracy.mean():>18.3f}"

        print(
            f"| {model:<13} | {batch_size:>10} | {max_epochs:>6} "
            f"| {accuracy_msg} | {runtime:>6.1f} Min "
            f"| {gpu_memory_usage:>8.1f} GByte |",
            flush=True,
        )
    print("-" * len(header))

    print(6, os.path.exists(save_to_model))
    # state_dict = torch.load(save_to_model)['model']
    # swav = swav.load_state_dict(state_dict)
    swav.eval()
    features = np.empty((NUM_TRAIN, FEATURES))
    labels = np.empty((NUM_TRAIN,), dtype=int)
    n = 0

    for images, label, file_name in dataloader_train_kNN:
        # print(file_name[0])
        labels[n:n + len(label)] = np.array(label)
        features[n:n + len(label)] = swav(images).detach().numpy()
        # r = swav(images).detach().numpy()
        # print(r)
        # print(r.shape)
        n += len(label)
        # break

    np.save(os.path.join(DIR_ROOT_SAVE, save_to, 'feature.npy'), features)
    np.save(os.path.join(DIR_ROOT_SAVE, save_to, 'labels.npy'), labels)
