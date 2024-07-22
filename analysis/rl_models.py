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


def make_batch(ses_data):
    batched_stims = []
    batched_full_rwd = []
    batched_choices = []
    batched_stimF0 = []
    batched_stimF1 = []
    batched_valid_mask = []
    for subj_idx, subj_data in enumerate(ses_data):
        batched_stims.append(subj_data['stim'])
        batched_full_rwd.append(subj_data['full_reward'])
        batched_choices.append(subj_data['choice'])
        batched_valid_mask.append(subj_data['choice'] >= 0)

        stimF1 = subj_data['stim']//setup.EXPERIMENT_SETUP['num_vals']
        stimF2 = subj_data['stim'] % setup.EXPERIMENT_SETUP['num_vals']

        if subj_data['infDimension'] == 1:
            batched_stimF0.append(stimF1)
            batched_stimF1.append(stimF2)
        else:
            batched_stimF0.append(stimF2)
            batched_stimF1.append(stimF1)

    return jnp.stack(batched_stims), jnp.stack(batched_stimF0), jnp.stack(batched_stimF1), \
        jnp.stack(batched_full_rwd), jnp.stack(batched_choices), jnp.stack(batched_valid_mask)


'''
individual subject likelihood
'''


def feature_based_model_subj(params, stims, stimsF0, stimsF1, full_rwd, choices, valid_mask, model_config, save_values=False):
    # load trial info
    num_subj = stims.shape[0]
    num_trials = stims.shape[1]
    num_vals = setup.EXPERIMENT_SETUP['num_vals']

    # load model info
    param2ind = model_config['param_indices']

    # initialize trial info
    # 2 features, each with 4 values
    init_values = 0.5*jnp.ones((num_subj, 2, num_vals))

    def trial_func(values, idx_trial):
        stim_f0 = stimsF0[:, idx_trial]  # num_subj X 2 options
        stim_f0 = nn.one_hot(stim_f0, num_classes=num_vals) # num_subj X 2 options X 4 values
        stim_f1 = stimsF1[:, idx_trial]
        stim_f1 = nn.one_hot(stim_f1, num_classes=num_vals) # num_subj X 2 options X 4 values
        valid_trial_mask = valid_mask[:, idx_trial]

        dv = params[:, param2ind['beta_attn']]*(values[:, 0]*stim_f0[:, 1]-values[:, 0]*stim_f0[:, 0]).sum(-1) + \
             params[:, param2ind['beta_noattn']]*(values[:, 1]*stim_f1[:, 1]-values[:, 1]*stim_f1[:, 0]).sum(-1) + \
             params[:, param2ind['bias']]

        dv = (1-valid_trial_mask)*0 + valid_trial_mask*dv

        with numpyro.plate('subj_trial', num_subj):
            if choices is not None:
                curr_ch = numpyro.sample('choice', dist.Bernoulli(logits=dv), obs=choices[:, idx_trial])
            else:
                curr_ch = numpyro.sample('choice', dist.Bernoulli(logits=dv))

        curr_unch = jnp.where(valid_trial_mask<0.5, -jnp.ones_like(curr_ch), 1-curr_ch)

        rw_ch = (1-valid_trial_mask)*0 + valid_trial_mask * full_rwd[jnp.arange(num_subj), idx_trial, curr_ch]  # num_subj
        rw_unch = (1-valid_trial_mask)*0 + valid_trial_mask * full_rwd[jnp.arange(num_subj), idx_trial, curr_unch]  # num_subj

        # all num_subj X 4
        stim_f0_ch = (stim_f0*nn.one_hot(curr_ch, num_classes=2)[..., None]).sum(1)
        stim_f0_unch = (stim_f0*nn.one_hot(curr_unch, num_classes=2)[..., None]).sum(1)
        stim_f1_ch = (stim_f1*nn.one_hot(curr_ch, num_classes=2)[..., None]).sum(1)
        stim_f1_unch = (stim_f1*nn.one_hot(curr_unch, num_classes=2)[..., None]).sum(1)
        stim_f0_decay = jnp.ones_like(stim_f0_ch) - stim_f0_ch - stim_f0_unch
        stim_f1_decay = jnp.ones_like(stim_f1_ch) - stim_f1_ch - stim_f1_unch

        # update manipulated feature of chosen option
        value_updates = jnp.zeros_like(values)
        value_updates = value_updates.at[:, 0, :].add(
            rw_ch[:, None] * params[:, param2ind['alpha_attn_ch_+']][:, None] * (1-values[:, 0]) * stim_f0_ch +
            (1-rw_ch[:, None]) * params[:, param2ind['alpha_attn_ch_-']][:, None] * (0-values[:, 0]) * stim_f0_ch)
        value_updates = value_updates.at[:, 1, :].add(
            rw_ch[:, None] * params[:, param2ind['alpha_noattn_ch_+']][:, None] * (1-values[:, 1]) * stim_f1_ch +
            (1-rw_ch[:, None]) * params[:, param2ind['alpha_noattn_ch_-']][:, None] * (0-values[:, 1]) * stim_f1_ch)

        value_updates = value_updates.at[:, 0, :].add(
            rw_unch[:, None] * params[:, param2ind['alpha_attn_unch_+']][:, None] * (1-values[:, 0]) * stim_f0_unch +
            (1-rw_unch[:, None]) * params[:, param2ind['alpha_attn_unch_-']][:, None] * (0-values[:, 0]) * stim_f0_unch)
        value_updates = value_updates.at[:, 1, :].add(
            rw_unch[:, None] * params[:, param2ind['alpha_noattn_unch_+']][:, None] * (1-values[:, 1]) * stim_f1_unch +
            (1-rw_unch[:, None]) * params[:, param2ind['alpha_noattn_unch_-']][:, None] * (0-values[:, 1]) * stim_f1_unch)

        # decay of unavailable options
        value_updates = value_updates.at[:, 0].add(
            params[:, param2ind['decay']][:, None] * (0.5-values[:, 0]) * stim_f0_decay)
        value_updates = value_updates.at[:, 1].add(
            params[:, param2ind['decay']][:, None] * (0.5-values[:, 1]) * stim_f1_decay)

        value_updates = (1-valid_trial_mask.reshape((-1, 1, 1))) * \
            0 + value_updates*valid_trial_mask.reshape((-1, 1, 1))

        if save_values:
            values = numpyro.deterministic("values", values+value_updates)
        else:
            values = values+value_updates

        return values, dv

    _, _ = scan(trial_func, init_values, jnp.arange(num_trials))
    return


def object_based_model_subj(params, stims, full_rwd, choices, valid_mask, model_config, save_values=False):
    # load trial info
    num_subj = stims.shape[0]
    num_trials = stims.shape[1]
    num_vals = setup.EXPERIMENT_SETUP['num_vals']

    # load model info
    param2ind = model_config['param_indices']

    # initialize trial info
    init_values = 0.5*jnp.ones((num_subj, num_vals**2))  # 16 objects

    def trial_func(values, idx_trial):
        stim = stims[:, idx_trial]
        # num_subj X 2 options X 16 values
        stim = nn.one_hot(stim, num_classes=num_vals**2)
        valid_trial_mask = valid_mask[:, idx_trial]

        dv = params[:, param2ind['beta']] * (values*stim[:, 1]-values*stim[:, 0]).sum(-1) + \
             params[:, param2ind['bias']]

        dv = (1-valid_trial_mask)*0 + valid_trial_mask*dv

        with numpyro.plate('subj_trial', num_subj):
            if choices is not None:
                curr_ch = numpyro.sample('choice', dist.Bernoulli(logits=dv), obs=choices[:, idx_trial])
            else:
                curr_ch = numpyro.sample('choice', dist.Bernoulli(logits=dv))

        curr_unch = jnp.where(valid_trial_mask<0.5, -jnp.ones_like(curr_ch), 1-curr_ch)

        rw_ch = (1-valid_trial_mask)*0 + valid_trial_mask * full_rwd[jnp.arange(num_subj), idx_trial, curr_ch]  # num_subj
        rw_unch = (1-valid_trial_mask)*0 + valid_trial_mask * full_rwd[jnp.arange(num_subj), idx_trial, curr_unch]  # num_subj

        # all num_subj X 16
        stim_ch = (stim*nn.one_hot(curr_ch, num_classes=2)[..., None]).sum(1)
        stim_unch = (stim*nn.one_hot(curr_unch, num_classes=2)[..., None]).sum(1)
        stim_decay = jnp.ones_like(stim_ch) - stim_ch - stim_unch

        # update manipulated feature of chosen option
        value_updates = jnp.zeros_like(values)
        value_updates = value_updates + \
            rw_ch[:, None] * params[:, param2ind['alpha_ch_+']][:, None] * (1-values) * stim_ch +\
            (1-rw_ch[:, None]) * params[:, param2ind['alpha_ch_-']][:, None] * (0-values) * stim_ch

        value_updates = value_updates + \
            rw_unch[:, None] * params[:, param2ind['alpha_unch_+']][:, None] * (1-values) * stim_unch +\
            (1-rw_unch[:, None]) * params[:, param2ind['alpha_unch_-']][:, None] * (0-values) * stim_unch

        # decay of unavailable options
        value_updates = value_updates + \
            params[:, param2ind['decay']][:, None] * (0.5-values) * stim_decay

        value_updates = (1-valid_trial_mask.reshape((-1, 1)))*0 + value_updates*valid_trial_mask.reshape((-1, 1))

        if save_values:
            values = numpyro.deterministic("values", values+value_updates)
        else:
            values = values+value_updates

        return values, dv

    _, _ = scan(trial_func, init_values, jnp.arange(num_trials))
    return


'''
(main) group level model specification
'''


def feature_based_model_group_ACL(stims, stimsF0, stimsF1, full_rwd, choices, valid_mask, save_values=False):
    model_config = {}
    model_config['param_names'] = [
        'bias', 'beta_attn', 'beta_noattn',
        'alpha_attn_ch_+', 'alpha_attn_ch_-',
        'alpha_noattn_ch_+', 'alpha_noattn_ch_-',
        'alpha_attn_unch_+', 'alpha_attn_unch_-',
        'alpha_noattn_unch_+', 'alpha_noattn_unch_-',
        'decay']
    model_config['param_indices'] = dict(
        zip(model_config['param_names'], list(range(len(model_config['param_names'])))))
    bias_indices = jnp.arange(0, 1)
    beta_indices = jnp.arange(1, 3)
    alpha_indices = jnp.arange(3, 12)
    num_params = 1 + len(beta_indices) + len(alpha_indices)
    num_subjects = len(stims)

    mu_prior_scale = jnp.ones(num_params)
    mu_prior_scale = mu_prior_scale.at[bias_indices].set(10)
    mu_prior_loc = jnp.zeros(num_params)
    # mean of bias (support real numbers)
    mu = numpyro.sample("mu", dist.Normal(mu_prior_loc, mu_prior_scale))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(10*jnp.ones(num_params)))

    with numpyro.plate("subj", num_subjects), numpyro.handlers.reparam(config={"subj_params": TransformReparam()}):
        subj_param_dist = dist.TransformedDistribution(
            dist.MultivariateNormal(
                jnp.zeros(num_params), jnp.eye(num_params)),
            dist.transforms.AffineTransform(mu, sigma))
        subj_params = numpyro.sample("subj_params", subj_param_dist)

    subj_params = subj_params.at[:, beta_indices].set(
        jnp.exp(subj_params[:, beta_indices]))
    subj_params = subj_params.at[:, alpha_indices].set(
        phi(subj_params[:, alpha_indices]))

    feature_based_model_subj(params=subj_params, stims=stims,
                             stimsF0=stimsF0, stimsF1=stimsF1,
                             full_rwd=full_rwd, choices=choices,
                             valid_mask=valid_mask,
                             model_config=model_config,
                             save_values=save_values)


def feature_based_model_group_AC(stims, stimsF0, stimsF1, full_rwd, choices, valid_mask, save_values=False):
    model_config = {}
    model_config['param_names'] = [
        'bias', 'beta_attn', 'beta_noattn',
        'alpha_attn_ch_+', 'alpha_attn_ch_-',
        'alpha_noattn_ch_+', 'alpha_noattn_ch_-',
        'alpha_attn_unch_+', 'alpha_attn_unch_-',
        'alpha_noattn_unch_+', 'alpha_noattn_unch_-',
        'decay'
    ]
    model_config['param_indices'] = dict(zip(model_config['param_names'],
                                             [0, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6, 7]))
    bias_indices = jnp.arange(0, 1)
    beta_indices = jnp.arange(1, 3)
    alpha_indices = jnp.arange(3, 8)
    num_params = 8
    num_subjects = len(stims)

    mu_prior_scale = jnp.ones(num_params)
    mu_prior_scale = mu_prior_scale.at[bias_indices].set(10)
    mu_prior_loc = jnp.zeros(num_params)
    # mean of bias (support real numbers)
    mu = numpyro.sample("mu", dist.Normal(mu_prior_loc, mu_prior_scale))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(10*jnp.ones(num_params)))

    with numpyro.plate("subj", num_subjects), numpyro.handlers.reparam(config={"subj_params": TransformReparam()}):
        subj_param_dist = dist.TransformedDistribution(
            dist.MultivariateNormal(
                jnp.zeros(num_params), jnp.eye(num_params)),
            dist.transforms.AffineTransform(mu, sigma))
        subj_params = numpyro.sample("subj_params", subj_param_dist)

    subj_params = subj_params.at[:, beta_indices].set(
        jnp.exp(subj_params[:, beta_indices]))
    subj_params = subj_params.at[:, alpha_indices].set(
        phi(subj_params[:, alpha_indices]))

    feature_based_model_subj(params=subj_params, stims=stims,
                             stimsF0=stimsF0, stimsF1=stimsF1,
                             full_rwd=full_rwd, choices=choices,
                             valid_mask=valid_mask,
                             model_config=model_config,
                             save_values=save_values)


def feature_based_model_group_AL(stims, stimsF0, stimsF1, full_rwd, choices, valid_mask, save_values=False):
    model_config = {}
    model_config['param_names'] = [
        'bias', 'beta_attn', 'beta_noattn',
        'alpha_attn_ch_+', 'alpha_attn_ch_-',
        'alpha_noattn_ch_+', 'alpha_noattn_ch_-',
        'alpha_attn_unch_+', 'alpha_attn_unch_-',
        'alpha_noattn_unch_+', 'alpha_noattn_unch_-',
        'decay'
    ]
    model_config['param_indices'] = dict(zip(model_config['param_names'],
                                             [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    bias_indices = jnp.arange(0, 1)
    beta_indices = jnp.arange(1, 2)
    alpha_indices = jnp.arange(2, 11)
    num_params = 11
    num_subjects = len(stims)

    mu_prior_scale = jnp.ones(num_params)
    mu_prior_scale = mu_prior_scale.at[bias_indices].set(10)
    mu_prior_loc = jnp.zeros(num_params)
    # mean of bias (support real numbers)
    mu = numpyro.sample("mu", dist.Normal(mu_prior_loc, mu_prior_scale))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(10*jnp.ones(num_params)))

    with numpyro.plate("subj", num_subjects), numpyro.handlers.reparam(config={"subj_params": TransformReparam()}):
        subj_param_dist = dist.TransformedDistribution(
            dist.MultivariateNormal(
                jnp.zeros(num_params), jnp.eye(num_params)),
            dist.transforms.AffineTransform(mu, sigma))
        subj_params = numpyro.sample("subj_params", subj_param_dist)

    subj_params = subj_params.at[:, beta_indices].set(
        jnp.exp(subj_params[:, beta_indices]))
    subj_params = subj_params.at[:, alpha_indices].set(
        phi(subj_params[:, alpha_indices]))

    feature_based_model_subj(params=subj_params, stims=stims,
                             stimsF0=stimsF0, stimsF1=stimsF1,
                             full_rwd=full_rwd, choices=choices,
                             valid_mask=valid_mask,
                             model_config=model_config,
                             save_values=save_values)


def feature_based_model_group_UA(stims, stimsF0, stimsF1, full_rwd, choices, valid_mask, save_values=False):
    model_config = {}
    model_config['param_names'] = [
        'bias', 'beta_attn', 'beta_noattn',
        'alpha_attn_ch_+', 'alpha_attn_ch_-',
        'alpha_noattn_ch_+', 'alpha_noattn_ch_-',
        'alpha_attn_unch_+', 'alpha_attn_unch_-',
        'alpha_noattn_unch_+', 'alpha_noattn_unch_-',
        'decay'
    ]
    model_config['param_indices'] = dict(zip(model_config['param_names'],
                                             [0, 1, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6]))
    bias_indices = jnp.arange(0, 1)
    beta_indices = jnp.arange(1, 2)
    alpha_indices = jnp.arange(2, 7)
    num_params = 7
    num_subjects = len(stims)

    mu_prior_scale = jnp.ones(num_params)
    mu_prior_scale = mu_prior_scale.at[bias_indices].set(10)
    mu_prior_loc = jnp.zeros(num_params)
    mu = numpyro.sample("mu", dist.Normal(mu_prior_loc, mu_prior_scale))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(10*jnp.ones(num_params)))

    with numpyro.plate("subj", num_subjects), numpyro.handlers.reparam(config={"subj_params": TransformReparam()}):
        subj_param_dist = dist.TransformedDistribution(
            dist.MultivariateNormal(
                jnp.zeros(num_params), jnp.eye(num_params)),
            dist.transforms.AffineTransform(mu, sigma))
        subj_params = numpyro.sample("subj_params", subj_param_dist)

    subj_params = subj_params.at[:, beta_indices].set(
        jnp.exp(subj_params[:, beta_indices]))
    subj_params = subj_params.at[:, alpha_indices].set(
        phi(subj_params[:, alpha_indices]))

    feature_based_model_subj(params=subj_params, stims=stims,
                             stimsF0=stimsF0, stimsF1=stimsF1,
                             full_rwd=full_rwd, choices=choices,
                             valid_mask=valid_mask,
                             model_config=model_config,
                             save_values=save_values)


def object_based_model_group(stims, stimsF0, stimsF1, full_rwd, choices, valid_mask, save_values=False):
    model_config = {}
    model_config['param_names'] = [
        'bias', 'beta',
        'alpha_ch_+', 'alpha_ch_-',
        'alpha_unch_+', 'alpha_unch_-',
        'decay']
    model_config['param_indices'] = dict(
        zip(model_config['param_names'], list(range(len(model_config['param_names'])))))
    bias_indices = jnp.arange(0, 1)
    beta_indices = jnp.arange(1, 2)
    alpha_indices = jnp.arange(2, 7)
    num_params = 1 + len(beta_indices) + len(alpha_indices)
    num_subjects = len(stims)

    mu_prior_scale = jnp.ones(num_params)
    mu_prior_scale = mu_prior_scale.at[bias_indices].set(10)
    mu_prior_loc = jnp.zeros(num_params)
    mu = numpyro.sample("mu", dist.Normal(mu_prior_loc, mu_prior_scale))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(10*jnp.ones(num_params)))

    with numpyro.plate("subj", num_subjects), numpyro.handlers.reparam(config={"subj_params": TransformReparam()}):
        subj_param_dist = dist.TransformedDistribution(
            dist.MultivariateNormal(
                jnp.zeros(num_params), jnp.eye(num_params)),
            dist.transforms.AffineTransform(mu, sigma))
        subj_params = numpyro.sample("subj_params", subj_param_dist)

    subj_params = subj_params.at[:, beta_indices].set(
        jnp.exp(subj_params[:, beta_indices]))
    subj_params = subj_params.at[:, alpha_indices].set(
        phi(subj_params[:, alpha_indices]))

    object_based_model_subj(params=subj_params, stims=stims,
                            full_rwd=full_rwd, choices=choices,
                            valid_mask=valid_mask,
                            model_config=model_config,
                            save_values=save_values)


'''
(ablated) group-level model specification
'''


def feature_based_model_group_AL_tied_chfg(stims, stimsF0, stimsF1, full_rwd, choices, valid_mask, save_values=False):
    model_config = {}
    model_config['param_names'] = [
        'bias', 'beta_attn', 'beta_noattn',
        'alpha_attn_ch_+', 'alpha_attn_ch_-',
        'alpha_noattn_ch_+', 'alpha_noattn_ch_-',
        'alpha_attn_unch_+', 'alpha_attn_unch_-',
        'alpha_noattn_unch_+', 'alpha_noattn_unch_-',
        'decay'
    ]
    model_config['param_indices'] = dict(zip(model_config['param_names'],
                                             [0, 1, 1, 2, 3, 4, 5, 3, 2, 5, 4, 6]))
    bias_indices = jnp.arange(0, 1)
    beta_indices = jnp.arange(1, 2)
    alpha_indices = jnp.arange(2, 7)
    num_params = 7
    num_subjects = len(stims)

    mu_prior_scale = jnp.ones(num_params)
    mu_prior_scale = mu_prior_scale.at[bias_indices].set(10)
    mu_prior_loc = jnp.zeros(num_params)
    mu = numpyro.sample("mu", dist.Normal(mu_prior_loc, mu_prior_scale))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(10*jnp.ones(num_params)))

    with numpyro.plate("subj", num_subjects), numpyro.handlers.reparam(config={"subj_params": TransformReparam()}):
        subj_param_dist = dist.TransformedDistribution(
            dist.MultivariateNormal(
                jnp.zeros(num_params), jnp.eye(num_params)),
            dist.transforms.AffineTransform(mu, sigma))
        subj_params = numpyro.sample("subj_params", subj_param_dist)

    subj_params = subj_params.at[:, beta_indices].set(
        jnp.exp(subj_params[:, beta_indices]))
    subj_params = subj_params.at[:, alpha_indices].set(
        phi(subj_params[:, alpha_indices]))

    feature_based_model_subj(params=subj_params, stims=stims,
                             stimsF0=stimsF0, stimsF1=stimsF1,
                             full_rwd=full_rwd, choices=choices,
                             valid_mask=valid_mask,
                             model_config=model_config,
                             save_values=save_values)


def feature_based_model_group_AL_tied_attn(stims, stimsF0, stimsF1, full_rwd, choices, valid_mask, save_values=False):
    model_config = {}
    model_config['param_names'] = [
        'bias', 'beta_attn', 'beta_noattn',
        'alpha_attn_ch_+', 'alpha_attn_ch_-',
        'alpha_noattn_ch_+', 'alpha_noattn_ch_-',
        'alpha_attn_unch_+', 'alpha_attn_unch_-',
        'alpha_noattn_unch_+', 'alpha_noattn_unch_-',
        'decay'
    ]
    model_config['param_indices'] = dict(zip(model_config['param_names'],
                                             [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    bias_indices = jnp.arange(0, 1)  # 0
    beta_indices = jnp.arange(1, 2)  # 1 for inverse temp,
    alpha_indices = jnp.arange(2, 7)  # 2,3,4,5 for learning rates, 6 for decay
    # 7 attention scaling (pre-probit transformation)
    scale_indices = jnp.arange(7, 8)
    num_params = 8
    num_subjects = len(stims)

    mu_prior_scale = jnp.ones(num_params)
    mu_prior_scale = mu_prior_scale.at[bias_indices].set(10)
    mu_prior_loc = jnp.zeros(num_params)
    mu = numpyro.sample("mu", dist.Normal(mu_prior_loc, mu_prior_scale))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(5*jnp.ones(num_params)))

    with numpyro.plate("subj", num_subjects), numpyro.handlers.reparam(config={"subj_params": TransformReparam()}):
        subj_param_dist = dist.TransformedDistribution(
            dist.MultivariateNormal(
                jnp.zeros(num_params), jnp.eye(num_params)),
            dist.transforms.AffineTransform(mu, sigma))
        subj_params = numpyro.sample("subj_params", subj_param_dist)

    subj_params_transformed = jnp.concatenate([
        subj_params[:, 0:1],
        jnp.exp(subj_params[:, 1:2]),
        phi(subj_params[:, 2:4]+subj_params[:, 7:8]), phi(subj_params[:, 2:4]),
        phi(subj_params[:, 4:6]+subj_params[:, 7:8]), phi(subj_params[:, 4:6]),
        phi(subj_params[:, 6:7]),
    ], axis=-1)

    feature_based_model_subj(params=subj_params_transformed, stims=stims,
                             stimsF0=stimsF0, stimsF1=stimsF1,
                             full_rwd=full_rwd, choices=choices,
                             valid_mask=valid_mask,
                             model_config=model_config,
                             save_values=save_values)


if __name__ == '__main__':
    processed_data_dir = "data/Processed"
    processed_file_name = "processed"
    figure_data_dir = "figures/All Processed"
    model_save_dir = "mcmc"
    # model_save_dir = "mcmc_ablated"
    # figure_data_dir = "figures/All Excluded"
    with open(os.path.join(processed_data_dir, processed_file_name), "rb") as f:
        data = pickle.load(f)

    print("starting mcmc fit")
    stims, stimsF0, stimsF1, full_rwd, choices, valid_mask = make_batch(data)

    fit_half = False
    if fit_half:
        model_save_dir = "mcmc_first_half"
        num_trials_to_fit = setup.EXPERIMENT_SETUP['numChoiceTrials']//2
        stims = stims[:,:num_trials_to_fit]
        stimsF0 = stimsF0[:,:num_trials_to_fit]
        stimsF1 = stimsF1[:,:num_trials_to_fit]
        full_rwd = full_rwd[:,:num_trials_to_fit]
        choices = choices[:,:num_trials_to_fit]
        valid_mask = valid_mask[:,:num_trials_to_fit]

    models_to_fit = [feature_based_model_group_ACL,
                     feature_based_model_group_AC,
                     feature_based_model_group_AL,
                     feature_based_model_group_UA,
                     object_based_model_group]
    model_names = ['F_ACL', 'F_AC', 'F_AL', 'F_UA', 'O']

    # models_to_fit = [
    #     feature_based_model_group_AL_tied_attn,
    #     feature_based_model_group_AL_tied_chfg,
    # ]
    # model_names = ['F_AL_tied_attn', 'F_AL_tied_chfg']

    all_model_mcmc = {}

    for m_name, m_func in zip(model_names, models_to_fit):

        nuts_kernel = NUTS(m_func, target_accept_prob=0.95,
                           init_strategy=init_to_median())

        mcmc = MCMC(nuts_kernel,
                    num_warmup=1000,
                    num_samples=2000,
                    num_chains=4,
                    progress_bar=True)
        key, subkey = random.split(key)
        mcmc.run(subkey, stims, stimsF0, stimsF1,
                 full_rwd, choices, valid_mask)

        inf_data = az.from_numpyro(mcmc)
        print(az.summary(inf_data, var_names=[
              'mu', 'sigma'], stat_focus='median', hdi_prob=0.95))
        print(az.waic(inf_data))

        all_model_mcmc[m_name] = mcmc

        with open(os.path.join(processed_data_dir, model_save_dir), 'wb') as f:
            pickle.dump(all_model_mcmc, f)
