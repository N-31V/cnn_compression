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
        name=task['model_name'],
        device='cuda:1'
    )
    exp.fit(p=fit_params)


def run_svd(task, fit_params, ft_params):
    for dec_mode in task['svd_params']['decomposing_mode']:
        for hoer_factor in task['svd_params']['hoer_loss_factor']:
            for orthogonal_factor in task['svd_params']['orthogonal_loss_factor']:
                exp = ClassificationExperimenter(
                    model=task['model'](**task['model_params']),
                    name=task['model_name'],
                    device='cuda:1'
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
                name=task['model_name'],
                device='cuda:1'
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
) -> None:
    train_ds, val_ds = task['dataset']()
    for f_params in task['fit_params']:
        fit_params = FitParameters(
            dataset_name=task['ds_name'],
            train_dl=DataLoader(train_ds, shuffle=True, **task['dataloader_params']),
            val_dl=DataLoader(val_ds, **task['dataloader_params']),
            **f_params
        )
        ft_params = FitParameters(
            dataset_name=task['ds_name'],
            train_dl=DataLoader(train_ds, shuffle=True, **task['dataloader_params']),
            val_dl=DataLoader(val_ds, **task['dataloader_params']),
            **task['ft_params']
        )
        MODS[mode](task, fit_params, ft_params)


print("Starting...")
start_t = datetime.now()
task = TASKS['ImageNet']
run(task)
run(task, mode='svd')
run(task, mode='sfp')
print(f'Total time: {datetime.now() - start_t}')
