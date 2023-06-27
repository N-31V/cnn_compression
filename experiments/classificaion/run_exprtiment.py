from datetime import datetime
from typing import Dict, Type, List
import logging

from torch.utils.data import DataLoader, Subset

from fedot_ind.core.architecture.experiment.nn_experimenter import ClassificationExperimenter, FitParameters
from fedot_ind.core.operation.optimization.structure_optimization import SFPOptimization, SVDOptimization
from exp_parameters import TASKS

logging.basicConfig(level=logging.INFO)

def run_base(task, fit_params, ft_params):
    exp = ClassificationExperimenter(
        model=task['model'](**task['model_params']),
        name=task['model_name']
    )
    exp.fit(p=fit_params)


def run_svd(task, fit_params, ft_params):
    for dec_mode in task['svd_params']['decomposing_mode']:
        for hoer_factor in task['svd_params']['hoer_loss_factor']:
            for orthogonal_factor in task['svd_params']['orthogonal_loss_factor']:
                exp = ClassificationExperimenter(
                    model=task['model'](**task['model_params']),
                    name=task['model_name']
                )
                svd_optim = SVDOptimization(
                    energy_thresholds=task['svd_params']['energy_thresholds'],
                    decomposing_mode=dec_mode,
                    hoer_loss_factor=hoer_factor,
                    orthogonal_loss_factor=orthogonal_factor
                )
                svd_optim.fit(exp=exp, params=fit_params, ft_params=ft_params)


def run_sfp(task, fit_params, ft_params):
    const_params = {k: v for k, v in task['sfp_params'].items() if k != 'zeroing'}
    for zeroing_mode, zeroing_params in task['sfp_params']['zeroing'].items():
        for zeroing_param in zeroing_params:
            exp = ClassificationExperimenter(
                model=task['model'](**task['model_params']),
                name=task['model_name']
            )
            sfp_optim = SFPOptimization(
                zeroing_mode=zeroing_mode,
                zeroing_mode_params=zeroing_param,
                **const_params
            )
            sfp_optim.fit(exp=exp, params=fit_params, ft_params=ft_params)


MODS = {'base': run_base, 'svd': run_svd, 'sfp': run_sfp}


def run(
        task: Dict,
        mode: str = 'base',
        folds: List[int] = [0, 1, 2, 3, 4],
) -> None:
    dataset, ds_folds = task['dataset']()
    for fold in folds:
        fold0 = Subset(dataset=dataset, indices=ds_folds[fold, 0, :])
        fold1 = Subset(dataset=dataset, indices=ds_folds[fold, 1, :])
        for i, train_ds, val_ds in [(0, fold0, fold1), (1, fold1, fold0)]:
            for f_params in task['fit_params']:
                fit_params = FitParameters(
                    dataset_name=task['ds_name'],
                    train_dl=DataLoader(train_ds, shuffle=True, **task['dataloader_params']),
                    val_dl=DataLoader(val_ds, **task['dataloader_params']),
                    description=f'{fold}_{i}',
                    **f_params
                )
                ft_params = FitParameters(
                    dataset_name=task['ds_name'],
                    train_dl=DataLoader(train_ds, shuffle=True, **task['dataloader_params']),
                    val_dl=DataLoader(val_ds, **task['dataloader_params']),
                    description=f'{fold}_{i}',
                    **task['ft_params']
                )
                MODS[mode](task, fit_params, ft_params)


f = [0, 1, 2, 3, 4]
# f = [0]
start_t = datetime.now()
for t in ['minerals200']:
    run(TASKS[t], folds=f)
    run(TASKS[t], mode='svd', folds=f)
    run(TASKS[t], mode='sfp', folds=f)
print(f'Total time: {datetime.now() - start_t}')
