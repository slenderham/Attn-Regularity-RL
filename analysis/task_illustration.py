import numpy as np
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd
from scipy import stats
from setup import EXPERIMENT_SETUP
from scipy.special import logit

import os
plt.rc('font', size=22, family='arial') 
plt.rc('axes', linewidth=2.5)
plt.rc('xtick.major', width=2, size=8)
plt.rc('ytick.major', width=2, size=8)
plt.rc('xtick.minor', width=1, size=4)
plt.rc('ytick.minor', width=1, size=4)


def plot_rwd_schedule_exp_var(figure_data_dir):

    rwd_schedule_for_fitting = np.stack([
        np.tile(np.arange(EXPERIMENT_SETUP['num_vals'])[None], (EXPERIMENT_SETUP['num_vals'], 1)).flatten(),
        np.tile(np.arange(EXPERIMENT_SETUP['num_vals'])[:,None], (1,EXPERIMENT_SETUP['num_vals'])).flatten(),
        np.arange(np.prod(EXPERIMENT_SETUP['reward_schedule'].shape)),
        logit(EXPERIMENT_SETUP['reward_schedule'].flatten())
    ], axis=-1)

    rwd_schedule_for_fitting = pd.DataFrame(rwd_schedule_for_fitting, columns=['F1', 'F2', 'O', 'p'])

    model = smf.ols('p~C(F1)+C(F2)', data=rwd_schedule_for_fitting).fit()

    ss_by_dimensions = sm.stats.anova_lm(model, typ=3)['sum_sq']
    eta_sq = ss_by_dimensions/np.sum(ss_by_dimensions)

    print(sm.stats.anova_lm(model, typ=3)['sum_sq'])

    plt.bar(np.arange(4), eta_sq, color=['grey', 'deepskyblue', 'darkorange', 'limegreen'])
    plt.gca().spines.right.set_visible(False)
    plt.gca().spines.top.set_visible(False)
    plt.xticks(np.arange(4), labels=['Mean', r'$F_m$', r'$F_n$', r'$O$'], fontsize=25)
    plt.xlabel('Dimension', fontsize=25)
    plt.ylabel('Prop. Var. Explained', fontsize=25)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_data_dir, "rwd_sch_var_exp.pdf"))
    plt.close()

def plot_est_error_by_weight(figure_data_dir):

    weight_scale = 1.5

    all_weights_to_try = np.logspace(-weight_scale, weight_scale, base=10)

    pO = EXPERIMENT_SETUP['reward_schedule']

    pF = pO.mean(0)[:,None]

    est_errs = []

    for weight in all_weights_to_try:
        pO_bar = pF.T*(pF**weight)/(pF.T*(pF**weight)+(1-pF.T)*((1-pF)**weight))

        est_errs.append(stats.pearsonr(logit(pO).flatten(), logit(pO_bar).flatten()).correlation)


    plt.plot(all_weights_to_try, est_errs, '-ok')
    plt.gca().spines.right.set_visible(False)
    plt.gca().spines.top.set_visible(False)
    plt.xscale('log')
    plt.xlabel('Feature weight', fontsize=25)
    plt.text(10**(-weight_scale)*2.5, plt.ylim()[1]*0.9, 'Underweigh', ha='center', 
             fontsize=20)
    plt.text(1, plt.ylim()[1]*1.0, 'Bayes optimal', ha='center', 
             fontsize=20)
    plt.text(10**weight_scale/2.5, plt.ylim()[1]*0.9, 'Overweigh', ha='center', 
             fontsize=20)
    plt.ylim(plt.ylim()[0], plt.ylim()[1]*1.05)
    plt.ylabel('Correlation', fontsize=25)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_data_dir, "est_error_by_weight.pdf"))
    plt.close()


if __name__=='__main__':
    figure_data_dir = "figures/All Processed"
    plot_rwd_schedule_exp_var(figure_data_dir)
    plot_est_error_by_weight(figure_data_dir)


    