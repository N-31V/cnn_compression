from typing import Dict, Callable
import os
import shutil
from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader

from fedot_ind.core.architecture.postprocessing.cv_results_parser import create_mean_exp
from fedot_ind.core.architecture.experiment.nn_experimenter import ClassificationExperimenter
from fedot_ind.core.operation.optimization.svd_tools import load_svd_state_dict
from fedot_ind.core.operation.optimization.structure_optimization import load_sfp_resnet_model

from optim.flops import flop
from experiments.classificaion.exp_parameters import ENERGY_ZEROING, PERSRNTAGE_ZEROIG, TASKS



def exp_list(dataset_name: str, folds=[0, 1],  n=[0, 1, 2, 3, 4]) -> Dict:
    config = TASKS[dataset_name]
    hoer_loss_factor = config['svd_params']['hoer_loss_factor']
    orthogonal_loss_factor = config['svd_params']['orthogonal_loss_factor']
    root = f"/media/n31v/data/results/{config['ds_name']}/{config['model_name']}"
    exps = {
        'Baseline': f"{root}/mean",
        'SFP energy': {f'SFP energy {e}': f'{root}_SFP_energy_threshold-{e}/mean' for e in ENERGY_ZEROING},
        'SFP percentage': {f'SFP {int(pr * 100)}%': f'{root}_SFP_pruning_ratio-{pr}/mean' for pr in PERSRNTAGE_ZEROIG},
        'SVD channel': {f'Hoer {hoer} Ort {ort}': f'{root}_SVD_channel_O-{ort}_H-{hoer}/mean' for hoer in hoer_loss_factor for ort in orthogonal_loss_factor},
        'SVD spatial': {f'Hoer {hoer} Ort {ort}': f'{root}_SVD_spatial_O-{ort}_H-{hoer}/mean' for hoer in hoer_loss_factor for ort in orthogonal_loss_factor},
    }
    return exps


def create_mean_exps(root, cond=lambda x: True):
    exps = os.listdir(root)
    for exp in filter(cond, exps):
        path = os.path.join(root, exp)
        try:
            create_mean_exp(path)
            print(path)
        except Exception as e:
            print(e)
    print('Done')


def del_mean_exps(root, cond=lambda x: True):
    exps = os.listdir(root)
    for exp in filter(cond, exps):
        path = os.path.join(root, exp, 'mean')
        try:
            shutil.rmtree(path)
            print(f'{path} removed.')
        except Exception as e:
            print(e)
    print('Done')


def compare_inference(
        dataset_name: str,
        exps: Dict,
        compare_dict: Dict,
        best_svd: Dict,
        percent: int = 99,
        sfp_load_fn: Callable = load_sfp_resnet_model
) -> pd.DataFrame:

    best_exps = {}
    for k, v in compare_dict.items():
        best_exps[k] = v.loc[v['fine-tuned'] >= percent]['size'].idxmin()

    task = TASKS[dataset_name]
    dataset, _ = task['dataset']()
    val_dl = DataLoader(dataset, **task['dataloader_params'])
    x, y = next(iter(val_dl))

    stats = {
        'size': {},
        'flop': {},
        'time': {},
        'f1': {}
    }

    for i in range(5):
        for j in range(2):
            fold = f'{i}_{j}'
            print(fold)
            for param in stats.values():
                param[fold] = {}

            models = {'Baseline': task['model'](**task['model_params'])}
            sd_path = os.path.join(os.path.split(exps['Baseline'])[0], fold, 'train.sd.pt')
            models['Baseline'].load_state_dict(torch.load(sd_path))
            stats['size'][fold]['Baseline'] = os.path.getsize(sd_path)

            for dec_mode in ['channel', 'spatial']:
                exp = exps[f'SVD {dec_mode}'][best_svd[dec_mode]]
                sd_path = os.path.join(os.path.split(exp)[0], fold, f"{best_exps[f'SVD {dec_mode}']}.sd.pt")
                for forward_mode in ['one_layer', 'two_layers', 'three_layers']:
                    model = task['model'](**task['model_params'])
                    load_svd_state_dict(model=model, decomposing_mode=dec_mode, state_dict_path=sd_path, forward_mode=forward_mode)
                    models[f'SVD {dec_mode} {forward_mode}'] = model
                    stats['size'][fold][f'SVD {dec_mode} {forward_mode}'] = os.path.getsize(sd_path)

            for sfp_mode in ['SFP percentage', 'SFP energy']:
                sd_path = os.path.join(os.path.split(exps[sfp_mode][best_exps[sfp_mode]])[0], fold, 'pruned.sd.pt')
                models[sfp_mode] = sfp_load_fn(sd_path)
                stats['size'][fold][sfp_mode] = os.path.getsize(sd_path)

            for name, model in models.items():
                f = flop(model, dataset_name, input_size=x.shape)
                stats['flop'][fold][name] = sum([v['flops'] for v in f.values()])

            for name, model in models.items():
                exp = ClassificationExperimenter(model=model)
                start_t = datetime.now()
                metrics = exp.val_loop(val_dl)
                print(f"{name}: {metrics['f1']}")
                stats['time'][fold][name] = datetime.now() - start_t
                stats['f1'][fold][name] = metrics['f1']

    for k, v in stats.items():
        tmp = pd.DataFrame(v)
        stats[k] = tmp.mean(axis=1)

    compare_df = pd.DataFrame(stats)

    for param in stats.keys():
        compare_df[f'{param}, %'] = compare_df[param] / compare_df.loc['Baseline', param] * 100

    return compare_df
