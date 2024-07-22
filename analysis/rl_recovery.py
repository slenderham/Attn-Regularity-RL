import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from numpyro.infer.reparam import TransformReparam
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan, cond
from jax import random
from jax import nn
import jax
from jax.scipy.special import ndtr as phi
import arviz as az

from matplotlib import pyplot as plt

import os
import pickle
import setup

numpyro.set_host_device_count(4)
key = random.PRNGKey(0)

from rl_models import *

if __name__=='__main__':
    processed_data_dir = "data/Processed"
    processed_file_name = "processed"
    figure_data_dir = "figures/All Processed"
    model_save_dir = "mcmc_recovery"
    with open(os.path.join(processed_data_dir, processed_file_name), "rb") as f:
        data = pickle.load(f)

    # load the trial data and empirical choices
    stims, stimsF0, stimsF1, full_rwd, choices, valid_mask = make_batch(data)
    
    # load the fitted models to get the population level posteriors
    with open(os.path.join(processed_data_dir, "mcmc"), 'rb') as f:
        all_models_mcmc_fit = pickle.load(f)

    models_to_fit = [feature_based_model_group_ACL,
                     feature_based_model_group_AC,
                     feature_based_model_group_AL,
                     feature_based_model_group_UA,
                     object_based_model_group]
    model_names = ['F_ACL', 'F_AC', 'F_AL', 'F_UA', 'O']
    
    # models_to_fit = [feature_based_model_group_AL]
    # model_names = ['F_AL']

    all_model_mcmc_recover = {} # tuple of generating model name and fitted model name

    num_subj, num_trials, _ = stims.shape

    print("starting mcmc fit")
    for gen_m_name, gen_m_func in zip(model_names, models_to_fit):
        print('==============================================')
        print(f'generating from {gen_m_name}')
        # use the posterior median of population mean and std
        gen_mu = jnp.median(all_models_mcmc_fit[gen_m_name].get_samples()['mu'], 0)
        gen_sigma = jnp.median(all_models_mcmc_fit[gen_m_name].get_samples()['sigma'], 0)

        # sample random parameters
        num_params = gen_mu.shape[0]
        key, subkey = random.split(key)
        gen_subj_params_base = random.normal(key=subkey, shape=(57, num_params))
        gen_subj_params_base = (gen_subj_params_base-gen_subj_params_base.mean(0, keepdims=True))/gen_subj_params_base.std(0, keepdims=True)
        gen_subj_params = gen_mu[None] + gen_sigma[None]*gen_subj_params_base

        gen_params = {
            'mu': gen_mu[None],
            'sigma': gen_sigma[None],
            'subj_params': gen_subj_params[None],
            'subj_params_base': gen_subj_params_base[None],
        }

        # make predictive model for sampling
        predictive_model = Predictive(gen_m_func, gen_params)
        key, subkey = random.split(key)
        predictive_gen = predictive_model(subkey, stims, stimsF0, stimsF1, full_rwd, None, valid_mask)
        
        # save choices, use the same sequence for all fit models
        gen_choices = predictive_gen['choice'].squeeze(0).T

        all_model_mcmc_recover[gen_m_name] = {
            'params': gen_params['subj_params'],
            'choices': gen_choices
        }

        for fit_m_name, fit_m_func in zip(model_names, models_to_fit):
            print(f'fitting from {fit_m_name}')
            nuts_kernel = NUTS(fit_m_func, target_accept_prob=0.95,
                            init_strategy=init_to_median())
        
            mcmc = MCMC(nuts_kernel, 
                        num_warmup=1000, 
                        num_samples=2000, 
                        num_chains=4,
                        progress_bar=True)
            key, subkey = random.split(key)
            mcmc.run(subkey, stims, stimsF0, stimsF1, full_rwd, gen_choices, valid_mask)
        
            inf_data = az.from_numpyro(mcmc)
            print(az.summary(inf_data, var_names=['mu', 'sigma'], stat_focus='median', hdi_prob=0.95))
            print(az.waic(inf_data))

            all_model_mcmc_recover[(gen_m_name, fit_m_name)] = mcmc

            with open(os.path.join(processed_data_dir, model_save_dir), 'wb') as f:
                pickle.dump(all_model_mcmc_recover, f)
    