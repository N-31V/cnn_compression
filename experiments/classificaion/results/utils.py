import os
import shutil

from fedot_ind.core.architecture.postprocessing.cv_results_parser import create_mean_exp

from experiments.classificaion.exp_parameters import ENERGY_ZEROING, PERSRNTAGE_ZEROIG, TASKS


def exp_list(dataset: str, folds=[0, 1],  n=[0, 1, 2, 3, 4]):
    config = TASKS[dataset]
    hoer_loss_factor = config['svd_params']['hoer_loss_factor']
    orthogonal_loss_factor = config['svd_params']['orthogonal_loss_factor']
    root = f"/media/n31v/data/results/{config['ds_name']}/{config['model_name']}"
    exps = {'Baseline': f"{root}/mean"}
    sfp_exps = {
        'energy': {f'SFP energy {e}': f'{root}_SFP_energy_threshold-{e}/mean' for e in ENERGY_ZEROING},
        'percentage': {f'SFP {int(pr * 100)}%': f'{root}_SFP_pruning_ratio-{pr}/mean' for pr in PERSRNTAGE_ZEROIG},
    }
    svd_exps = {
        'channel': {f'Hoer {hoer} Ort {ort}': f'{root}_SVD_channel_O-{ort}_H-{hoer}/mean' for hoer in hoer_loss_factor for ort in orthogonal_loss_factor},
        'spatial': {f'Hoer {hoer} Ort {ort}': f'{root}_SVD_spatial_O-{ort}_H-{hoer}/mean' for hoer in hoer_loss_factor for ort in orthogonal_loss_factor},
    }
    return exps, sfp_exps, svd_exps


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
