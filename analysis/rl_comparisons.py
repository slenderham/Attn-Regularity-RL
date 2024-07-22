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
import seaborn as sns
az.rcParams["plot.max_subplots"] = 200

from matplotlib import pyplot as plt

import os
import pickle
import setup

from rl_models import *

# TODO: compare F full and F with tied learning rates
# TODO: compare F_ACL, F_AC, F_AL, F_UA, O, F+O

def plot_main_comparison(processed_data_dir):

    with open(os.path.join(processed_data_dir, "mcmc"), 'rb') as f:
        all_models_mcmc = pickle.load(f)

    compare_dict = {}

    for k, v in all_models_mcmc.items():
        inf_data = az.from_numpyro(v)
        log_lik = inf_data.log_likelihood
        log_lik['choice_total'] = log_lik.sum('choice_dim_0').to_array()
        compare_dict[k] = az.waic(inf_data, var_name='choice_total')

    print('loaded all models')
    print('starting model comparisons')
    comp_df = az.compare(compare_dict, ic='waic')
    print(comp_df)

    plt.bar(np.arange(comp_df.shape[0]), comp_df.loc[:,'elpd_diff'], edgecolor='black', facecolor=[0.9,0.9,0.9], yerr=comp_df.loc[:,'dse'])
    plt.xticks(np.arange(comp_df.shape[0]), [f'{name}' for name in comp_df.index.values], rotation=20)
    plt.ylabel('$\Delta$ WAIC')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_data_dir, "rl_model_comparison.pdf"))
    plt.close()

    plt.bar(np.arange(comp_df.shape[0]), comp_df.loc[:,'weight'], edgecolor='black', facecolor=[0.9,0.9,0.9])
    plt.xticks(np.arange(comp_df.shape[0]), [f'{name}' for name in comp_df.index.values], rotation=20)
    plt.ylabel('Stacking weights')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_data_dir, "rl_weights.pdf"))
    plt.close()


def plot_ablation_comparison():





if __name__=='__main__':

    processed_data_dir = "data/Processed"
    processed_file_name = "processed"
    # processed_file_name = "excluded"
    figure_data_dir = "figures/All Processed"
    
    with open(os.path.join(processed_data_dir, "mcmc"), 'rb') as f:
        all_models_mcmc = pickle.load(f)

    comp_df = az.compare(all_models_mcmc, ic='waic')
    print(comp_df)

    az.plot_compare(comp_df)
    plt.show()
    # for model_name, model_mcmc in all_models_mcmc.items():
    #     inf_data = az.from_numpyro(model_mcmc)
    #     print(model_name, az.waic(inf_data))