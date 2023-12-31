from datetime import datetime
from typing import Dict, List, Optional
import logging

import torch.utils.data
from torch.utils.data import DataLoader, Subset

from fedot_ind.core.architecture.experiment.nn_experimenter import ClassificationExperimenter, FitParameters
from fedot_ind.core.operation.optimization.structure_optimization import SFPOptimization, SVDOptimization
from exp_parameters import TASKS

logging.basicConfig(level=logging.INFO)


def run_base(task, fit_params, ft_params):
    exp = ClassificationExperimenter(
        model=task['model'](**task['model_params']),
        name=task['model_name'],
        device=task['device']
    )
    exp.fit(p=fit_params)


def run_svd(task, fit_params, ft_params):
    for dec_mode in task['svd_params']['decomposing_mode']:
        for hoer_factor in task['svd_params']['hoer_loss_factor']:
            for orthogonal_factor in task['svd_params']['orthogonal_loss_factor']:
                exp = ClassificationExperimenter(
                    model=task['model'](**task['model_params']),
                    name=task['model_name'],
                    device=task['device']
                )
                svd_optim = SVDOptimization(
                    energy_thresholds=task['svd_params']['energy_thresholds'],
                    decomposing_mode=dec_mode,
                    hoer_loss_factor=hoer_factor,
                    orthogonal_loss_factor=orthogonal_factor
                )
                svd_optim.fit(exp=exp, params=fit_params, ft_params=ft_params)


def run_sfp(task, fit_params, ft_params):
    const_params = {k: v for k, v in task['sfp_params'].items() if k != 'zeroing_fn'}
    for zeroing_fn in task['sfp_params']['zeroing_fn']:
        exp = ClassificationExperimenter(
            model=task['model'](**task['model_params']),
            name=task['model_name'],
            device=task['device']
        )
        sfp_optim = SFPOptimization(zeroing_fn, **const_params)
        sfp_optim.fit(exp=exp, params=fit_params, ft_params=ft_params)


MODS = {'base': run_base, 'svd': run_svd, 'sfp': run_sfp}


def run(
        task: Dict,
        train_ds: Optional[torch.utils.data.Dataset] = None,
        val_ds: Optional[torch.utils.data.Dataset] = None,
        mode: str = 'base',
        description: str = '',
) -> None:
    if train_ds is None:
        train_ds, val_ds = task['dataset']()
    fit_params = FitParameters(
        dataset_name=task['ds_name'],
        train_dl=DataLoader(train_ds, shuffle=True, **task['dataloader_params']),
        val_dl=DataLoader(val_ds, **task['dataloader_params']),
        description=description,
        **task['fit_params']
    )
    ft_params = FitParameters(
        dataset_name=task['ds_name'],
        train_dl=DataLoader(train_ds, shuffle=True, **task['dataloader_params']),
        val_dl=DataLoader(val_ds, **task['dataloader_params']),
        description=description,
        **task['ft_params']
    )
    MODS[mode](task, fit_params, ft_params)


def run_with_folds(
        task: Dict,
        mode: str = 'base',
        folds: List[int] = [0, 1, 2, 3, 4],
) -> None:
    dataset, ds_folds = task['dataset']()
    for fold in folds:
        fold0 = Subset(dataset=dataset, indices=ds_folds[fold, 0, :])
        fold1 = Subset(dataset=dataset, indices=ds_folds[fold, 1, :])
        for i, train_ds, val_ds in [(0, fold0, fold1), (1, fold1, fold0)]:
            run(task, train_ds, val_ds, mode, description=f'{fold}_{i}')


if __name__ == '__main__':
    f = [0, 1, 2, 3, 4]
    tasks = ['MNIST', 'FashionMNIST', 'CIFAR10', 'LUSC', 'minerals']
    # tasks = ['minerals']
    for t in tasks:
        start_t = datetime.now()
        run_with_folds(TASKS[t], folds=f)
        run_with_folds(TASKS[t], mode='svd', folds=f)
        run_with_folds(TASKS[t], mode='sfp', folds=f)
        print(f'Total {t} time: {datetime.now() - start_t}')
