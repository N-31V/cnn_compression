import os
from typing import Optional, Tuple

import numpy as np
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision.datasets import ImageFolder, MNIST, CIFAR10, FashionMNIST, ImageNet
from torchvision.models import resnet18, resnet50, resnet101

from simple_conv_net import SimpleConvNet3, prune_simple_conv_net

DATASETS_ROOT = '/media/n31v/data/datasets'
RESULT_PATH = '/media/n31v/data/results'

SVD_PARAMS = {
    'energy_thresholds': [0.1, 0.3, 0.5, 0.7, 0.9, 0.93, 0.96, 0.99, 0.999],
    'decomposing_mode': ['channel', 'spatial'],
    'hoer_loss_factor': [0.1, 0.01, 0.001],
    'orthogonal_loss_factor': [1, 10, 100]
}
SFP_PARAMS = {
    'percentage': [{'pruning_ratio': p} for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]],
    'energy': [{'energy_threshold': e} for e in [0.9, 0.93, 0.96, 0.99, 0.999]],
}

MIDL_GROUP_PARAMS = {
    'dataloader_params': {'batch_size': 32, 'num_workers': 8},
    'model': resnet18,
    'model_name': 'ResNet18',
    'model_params': {'num_classes': 21},
    'fit_params': [
        {
            'num_epochs': 30,
            'lr_scheduler': StepLR,
            'lr_scheduler_params': {'step_size': 10, 'gamma': 0.2},
            'models_path': RESULT_PATH,
            'summary_path': RESULT_PATH,
        }
    ],
    'ft_params':
        {
            'num_epochs': 6,
            'lr_scheduler': StepLR,
            'lr_scheduler_params': {'step_size': 2, 'gamma': 0.2},
            'models_path': RESULT_PATH,
            'summary_path': RESULT_PATH,
        },
    'svd_params': SVD_PARAMS,
    'sfp_params': {
        'zeroing': SFP_PARAMS,
    }
}

def get_mnist():
    ds = MNIST(
        root=DATASETS_ROOT,
        transform=Compose([ToTensor(), Normalize(mean=(0.13066,), std=(0.3081,))])
    )
    folds = np.load(os.path.join(DATASETS_ROOT, 'MNIST', 'folds.npy'))
    return ds, folds


def get_fashion():
    ds = FashionMNIST(
        root=DATASETS_ROOT,
        transform=Compose([ToTensor(), Normalize(mean=(0.286,), std=(0.353,))])
    )
    folds = np.load(os.path.join(DATASETS_ROOT, 'FashionMNIST', 'folds.npy'))
    return ds, folds


def get_cifar():
    ds = CIFAR10(
        root=os.path.join(DATASETS_ROOT, 'CIFAR10'),
        transform=Compose([
            ToTensor(),
            Resize((28, 28), antialias=True),
            Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616))
        ])
    )
    folds = np.load(os.path.join(DATASETS_ROOT, 'CIFAR10', 'folds.npy'))
    return ds, folds


def get_imagenet():
    train_ds = ImageNet(
        root=os.path.join(DATASETS_ROOT, 'ImageNet'),
        transform=Compose([
            ToTensor(),
            Resize((500, 500), antialias=True),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    )
    val_ds = ImageNet(
        root=os.path.join(DATASETS_ROOT, 'ImageNet'),
        split='val',
        transform=Compose([
            ToTensor(),
            Resize((500, 500), antialias=True),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    )
    return train_ds, val_ds


def get_image_folder(dataset: str, transforms):
    def get_ds():
        ds = ImageFolder(
            root=os.path.join(DATASETS_ROOT, dataset),
            transform=transforms
        )
        folds = np.load(os.path.join(DATASETS_ROOT, dataset, 'folds.npy'))
        return ds, folds

    return get_ds


TEST_SVD_PARAMS = {
    'ds_name': 'LUSC_SVD',
    'dataset': get_image_folder(
        dataset='Land-Use_Scene_Classification/images',
        transforms=Compose([
            ToTensor(),
            Resize((200, 200), antialias=True),
            Normalize(mean=(0.462, 0.471, 0.440), std=(0.287, 0.279, 0.269))
        ])),
    'dataloader_params': {'batch_size': 32, 'num_workers': 8},
    'model_params': {'num_classes': 21},
    'fit_params': [
        {
            'num_epochs': 50,
            'optimizer_params': {'lr': 0.0005},
            'lr_scheduler': StepLR,
            'lr_scheduler_params': {'step_size': 15, 'gamma': 0.2},
            'models_path': RESULT_PATH,
            'summary_path': RESULT_PATH,
        }
    ],
    'ft_params':
        {
            'num_epochs': 10,
            'optimizer_params': {'lr': 0.0005},
            'lr_scheduler': StepLR,
            'lr_scheduler_params': {'step_size': 3, 'gamma': 0.2},
            'models_path': RESULT_PATH,
            'summary_path': RESULT_PATH,
        },
    'svd_params': {
        'energy_thresholds': [0.9, 0.99],
        'decomposing_mode': ['channel', 'spatial'],
        'hoer_loss_factor': [0.1],
        'orthogonal_loss_factor': [10]
    },
    'sfp_params': {}
}


TASKS = {

    'MNIST': {
        'ds_name': 'MNIST',
        'dataset': get_mnist,
        'dataloader_params': {'batch_size': 512, 'num_workers': 8},
        'model': SimpleConvNet3,
        'model_name': 'SimpleConvNet',
        'model_params': {'num_classes': 10, 'in_channels': 1},
        'fit_params': [{'num_epochs': 10, 'models_path': RESULT_PATH, 'summary_path': RESULT_PATH}],
        'ft_params': {'num_epochs': 3, 'models_path': RESULT_PATH, 'summary_path': RESULT_PATH},
        'svd_params': SVD_PARAMS,
        'sfp_params': {
            'zeroing': SFP_PARAMS,
            'final_pruning_fn': prune_simple_conv_net,
            'model_class': SimpleConvNet3,
        },
    },

    'FashionMNIST': {
        'ds_name': 'FashionMNIST',
        'dataset': get_fashion,
        'dataloader_params': {'batch_size': 512, 'num_workers': 8},
        'model': SimpleConvNet3,
        'model_name': 'SimpleConvNet',
        'model_params': {'num_classes': 10, 'in_channels': 1},
        'fit_params': [{'num_epochs': 10, 'models_path': RESULT_PATH, 'summary_path': RESULT_PATH}],
        'ft_params': {'num_epochs': 3, 'models_path': RESULT_PATH, 'summary_path': RESULT_PATH},
        'svd_params': SVD_PARAMS,
        'sfp_params': {
            'zeroing': SFP_PARAMS,
            'final_pruning_fn': prune_simple_conv_net,
            'model_class': SimpleConvNet3,
        },
    },

    'CIFAR10': {
        'ds_name': 'CIFAR10',
        'dataset': get_cifar,
        'dataloader_params': {'batch_size': 32, 'num_workers': 8},
        'model': SimpleConvNet3,
        'model_name': 'SimpleConvNet',
        'model_params': {'num_classes': 10},
        'fit_params': [
            {
                'num_epochs': 20,
                'lr_scheduler': StepLR,
                'lr_scheduler_params': {'step_size': 5, 'gamma': 0.2, 'verbose': True},
                'models_path': RESULT_PATH,
                'summary_path': RESULT_PATH,
            }
        ],
        'ft_params':
            {
                'num_epochs': 6,
                'lr_scheduler': StepLR,
                'lr_scheduler_params': {'step_size': 2, 'gamma': 0.2, 'verbose': True},
                'models_path': RESULT_PATH,
                'summary_path': RESULT_PATH,
            },
        'svd_params': SVD_PARAMS,
        'sfp_params': {
            'zeroing': SFP_PARAMS,
            'final_pruning_fn': prune_simple_conv_net,
            'model_class': SimpleConvNet3,
        },
    },


    'LUSC': {
        'ds_name': 'LUSC',
        'dataset': get_image_folder(
            dataset='Land-Use_Scene_Classification/images',
            transforms=Compose([
                ToTensor(),
                Resize((200, 200), antialias=True),
                Normalize(mean=(0.462, 0.471, 0.440), std=(0.287, 0.279, 0.269))
            ])),
        **MIDL_GROUP_PARAMS
    },

    'minerals200': {
        'ds_name': 'minerals200_10_02',
        'dataset': get_image_folder(
            dataset='minerals_21_200',
            transforms=Compose([
                ToTensor(),
                Resize((200, 200), antialias=True),
                Normalize(mean=(0.54, 0.61, 0.51), std=(0.22, 0.23, 0.23))
            ])),
        **MIDL_GROUP_PARAMS
    },


    'LUSC_resnet18': {
        **TEST_SVD_PARAMS,
        'model': resnet18,
        'model_name': 'ResNet18',
    },
    'LUSC_resnet50': {
        **TEST_SVD_PARAMS,
        'model': resnet50,
        'model_name': 'ResNet50',
    },


    'ImageNet': {
        'ds_name': 'ImageNet',
        'dataset': get_imagenet,
        'dataloader_params': {'batch_size': 16, 'num_workers': 8},
        'model': resnet50,
        'model_name': 'ResNet50',
        'model_params': {'num_classes': 1000},
        'fit_params': [
            {
                'num_epochs': 50,
                'optimizer_params': {'lr': 0.0005},
                'lr_scheduler': StepLR,
                'lr_scheduler_params': {'step_size': 15, 'gamma': 0.2},
                'models_path': RESULT_PATH,
                'summary_path': RESULT_PATH,
            }
        ],
        'ft_params':
            {
                'num_epochs': 10,
                'optimizer_params': {'lr': 0.0005},
                'lr_scheduler': StepLR,
                'lr_scheduler_params': {'step_size': 3, 'gamma': 0.2},
                'models_path': RESULT_PATH,
                'summary_path': RESULT_PATH,
            },
        'svd_params': SVD_PARAMS,
        'sfp_params': {
            'zeroing': SFP_PARAMS,
        },
    },
}
