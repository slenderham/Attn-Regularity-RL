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
az.rcParams["plot.max_subplots"] = 200

from matplotlib import pyplot as plt

import os
import pickle
import setup

from rl_models import *

if __name__=='__main__':

    processed_data_dir = "data/Processed"
    processed_file_name = "processed"
    # processed_file_name = "excluded"
    figure_data_dir = "figures/All Processed"
    
    with open(os.path.join(processed_data_dir, "mcmc"), 'rb') as f:
        all_models_mcmc = pickle.load(f)

    mcmc = all_models_mcmc['F+O_static']
    inf_data = az.from_numpyro(mcmc)
    print(inf_data)
    print(az.waic(inf_data))

    plots_to_make = ["trace", "ess", "rank", "pair"]
    # plots_to_make = ['forest', 'diff_posterior']

    if "waic" in plots_to_make:
        data_loo = az.loo(inf_data, pointwise=True)
        az.plot_khat(data_loo, show_bins=True)
        # plt.plot(data_loo.values[6].values.sum(0), 'k')
        plt.show()

    if "trace" in plots_to_make:
        az.plot_trace(inf_data, var_names=["mu", "sigma"])
        plt.tight_layout()
        plt.savefig(os.path.join(figure_data_dir, "rl_trace.pdf"))

    if "ess" in plots_to_make:
        az.plot_ess(inf_data, var_names=["mu", "sigma"], kind="evolution")
        plt.tight_layout()
        plt.savefig(os.path.join(figure_data_dir, "rl_ess.pdf"))

    if "rank" in plots_to_make:
        az.plot_rank(inf_data, var_names=["mu", "sigma"])
        plt.tight_layout()
        plt.savefig(os.path.join(figure_data_dir, "rl_rank.pdf"))

    if "pair" in plots_to_make:
        az.plot_pair(inf_data, group='posterior', var_names=["mu"], kind="kde", 
                    marginals=True, divergences=True,  
                    kde_kwargs={
                        "hdi_probs": [0.25, 0.5, 0.75],  # Plot 30%, 60% and 90% HDI contours
                        "contourf_kwargs": {"cmap": "Blues"},
                    })
        plt.tight_layout()
        plt.savefig(os.path.join(figure_data_dir, "rl_pair_mu.pdf"))

        az.plot_pair(inf_data, group='posterior', var_names=["sigma"], kind="kde", 
                    marginals=True, divergences=True,
                    kde_kwargs={
                        "hdi_probs": [0.25, 0.5, 0.75],  # Plot 30%, 60% and 90% HDI contours
                        "contourf_kwargs": {"cmap": "Blues"},
                    })
        plt.tight_layout()
        plt.savefig(os.path.join(figure_data_dir, "rl_pair_sigma.pdf"))

    if "forest" in plots_to_make:
        fig, axes = plt.subplots(1,6,figsize=(12,4))
        post = inf_data.posterior
        post['beta0'] = np.exp(post['mu'][...,1])
        post['beta1'] = np.exp(post['mu'][...,1])
        # post['beta'] = np.exp(post['mu'][...,3])
        
        post['alpha+_c,0'] = special.expit(post['mu'][...,4])
        post['alpha+_c,1'] = special.expit(post['mu'][...,6])
        post['alpha-_c,0'] = special.expit(post['mu'][...,5])
        post['alpha-_c,1'] = special.expit(post['mu'][...,7])
        
        post['alpha+_f,0'] = special.expit(post['mu'][...,8])
        post['alpha+_f,1'] = special.expit(post['mu'][...,10])
        post['alpha-_f,0'] = special.expit(post['mu'][...,9])
        post['alpha-_f,1'] = special.expit(post['mu'][...,11])

        # post['alpha+_c'] = special.expit(post['mu'][...,8])
        # post['alpha-_c'] = special.expit(post['mu'][...,9])
        # post['alpha+_f'] = special.expit(post['mu'][...,14])
        # post['alpha-_f'] = special.expit(post['mu'][...,15])

        post['d'] = special.expit(post['mu'][...,12])

        az.plot_violin(
            inf_data,
            var_names=["beta0"],
            side="left",
            ax=axes[0],
            hdi_prob=0.95,
        )
        az.plot_violin(
            inf_data,
            var_names=["beta1"],
            side="right",
            ax=axes[0],
            hdi_prob=0.95,
        )
        # az.plot_violin(
        #     inf_data,
        #     var_names=["beta"],
        #     ax=axes[1,0],
        #     hdi_prob=0.95,
        #     shade_kwargs={'color': 'green'}
        # )
        axes[0].set_title(r'$\beta$')
        az.plot_violin(
            inf_data,
            var_names=['alpha+_c,0', 'alpha-_c,0', 
                       'alpha+_f,0', 'alpha-_f,0'],
            side="left",
            sharey=False,
            sharex=False,
            ax=axes[1:5],
            hdi_prob=0.95,
        )
        az.plot_violin(
            inf_data,
            var_names=['alpha+_c,1', 'alpha-_c,1', 
                       'alpha+_f,1', 'alpha-_f,1'],
            side="right",
            sharey=False,
            sharex=False,
            ax=axes[1:5],
            hdi_prob=0.95,
        )
        # az.plot_violin(
        #     inf_data,
        #     var_names=['alpha+_c', 'alpha-_c', 
        #                'alpha+_f', 'alpha-_f'],
        #     sharey=False,
        #     sharex=False,
        #     ax=axes[1,1:5],
        #     hdi_prob=0.95,
        #     shade_kwargs={'color': 'green'}
        # )
        axes[1].set_title(r'$\alpha_{+, chosen}$')
        axes[2].set_title(r'$\alpha_{-, chosen}$')
        axes[3].set_title(r'$\alpha_{+, forgone}$')
        axes[4].set_title(r'$\alpha_{-, forgone}$')
        for a in axes[1:5]:
            a.set_ylim([-0.01,0.55])
        # for a in axes[1,1:5]:
        #     a.set_ylim([-0.01,0.55])
        # for a in axes[1]:
        #     a.set_title('')
        az.plot_violin(
            inf_data,
            var_names=["d"],
            ax=axes[5],
            hdi_prob=0.95,
        )
        axes[5].set_title(r'$d$')

        axes[0].set_ylabel('Feature')
        # axes[1,0].set_ylabel('Object')

        plt.tight_layout()
        plt.savefig(os.path.join(figure_data_dir, "rl_param_forest.pdf"))

    if "diff_posterior" in plots_to_make:
        post = inf_data.posterior
        # post['diff_beta'] = np.exp(post['mu'][...,1])-np.exp(post['mu'][...,2])
        post['diff_ch_r'] = special.expit(post['mu'][...,4])-special.expit(post['mu'][...,6])
        post['diff_ch_nr'] = special.expit(post['mu'][...,5])-special.expit(post['mu'][...,7])
        post['diff_fg_r'] = special.expit(post['mu'][...,10])-special.expit(post['mu'][...,12])
        post['diff_fg_nr'] = special.expit(post['mu'][...,11])-special.expit(post['mu'][...,13])
        # post['diff_decay'] = special.expit(post['mu'][...,11])-special.expit(post['mu'][...,12])
        
        az.plot_posterior(inf_data, var_names=['diff_ch_r', 'diff_ch_nr', 
                                               'diff_fg_r', 'diff_fg_nr'], 
                          hdi_prob=0.95, ref_val=0)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_data_dir, "rl_param_diff.pdf"))