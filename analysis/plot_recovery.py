import jax.numpy as jnp
import numpy as np
from scipy import special
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan, cond
from jax import random
from jax import nn
import jax
import arviz as az
from matplotlib import pyplot as plt
import pandas as pd
az.rcParams["plot.max_subplots"] = 200
plt.rcParams.update({'font.size': 20})

from matplotlib import pyplot as plt

import os
import pickle
import setup

import seaborn as sns
from rl_models import *

from utils import plot_mean_hpdi, nanmovmean

def plot_model_recovery(all_models_mcmc_recovery):
    label_mapper = {
        'F_ACL': f'$F_{{ACL}}$',
        'F_AC': f'$F_{{AC}}$',
        'F_AL': f'$F_{{AL}}$',
        'F_UA': f'$F$',
        'O': f'$O$',
    }

    compare_dict = {
        gen_m_name: {} for gen_m_name in label_mapper.keys()
    }

    for k, v in all_models_mcmc_recovery.items():
        if len(k)!=2: 
            continue
        inf_data = az.from_numpyro(v)
        log_lik = inf_data.log_likelihood
        log_lik['choice_total'] = log_lik.sum('choice_dim_0').to_array()
        compare_dict[k[0]][k[1]] = az.waic(inf_data, var_name='choice_total')

    elpd_diff = np.empty((5, 5))*np.nan
    elpd_diff_ses = np.empty((5, 5))*np.nan

    for gen_m_idx, gen_m in enumerate(label_mapper.keys()):
        comp_df = az.compare(compare_dict[gen_m], ic='waic')

        elpd_diff[gen_m_idx] = comp_df.loc[label_mapper.keys(),'elpd_diff']
        elpd_diff_ses[gen_m_idx] = comp_df.loc[label_mapper.keys(),'dse']

    elpd_annot = []

    for gen_m_idx in range(len(label_mapper.keys())):
        elpd_annot.append([])
        for fit_m_idx in range(len(label_mapper.keys())):
            elpd_annot[-1].append(f'{np.round(elpd_diff[gen_m_idx, fit_m_idx], 2)}$\pm${np.round(elpd_diff_ses[gen_m_idx, fit_m_idx], 2)}')
        
    sns.heatmap(elpd_diff, vmin=0, vmax=10, cmap='magma', annot=elpd_annot)
    plt.title('$\Delta$ WAIC')
    plt.savefig(os.path.join(figure_data_dir, "rl_model_recovery.pdf"))
    


if __name__=='__main__':
    processed_data_dir = "data/Processed"
    processed_file_name = "processed"
    figure_data_dir = "figures/All Processed"
    # model_save_dir = "mcmc_ablated"
    # figure_data_dir = "figures/All Excluded"
    with open(os.path.join(processed_data_dir, processed_file_name), "rb") as f:
        data = pickle.load(f)

    with open(os.path.join(processed_data_dir, "mcmc_recovery"), 'rb') as f:
        all_models_mcmc_recovery = pickle.load(f)

    plot_model_recovery(all_models_mcmc_recovery)

    