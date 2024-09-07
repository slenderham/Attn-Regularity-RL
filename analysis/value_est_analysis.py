import numpy as np
import statsmodels.api as sm
import statsmodels.stats as smstats
import pandas as pd
import os
import pickle
import setup as setup
import statsmodels.formula.api as smf
from scipy.special import logit
from matplotlib import pyplot as plt
plt.rc('font', size=25, family='arial') 
plt.rc('axes', linewidth=2.5)
plt.rc('xtick.major', width=2, size=8)
plt.rc('ytick.major', width=2, size=8)


def plot_value_error_curve(data, figure_data_dir):
    num_subj = len(data)
    perf_all = []

    pO = setup.EXPERIMENT_SETUP['reward_schedule']

    num_est_trials = len(setup.EXPERIMENT_SETUP['locEstimationTrials'])

    for idx_subj, sesdata in enumerate(data):
        curr_subj_perf = []
        for idx_trial in range(num_est_trials):
            stimOs = sesdata['est_stims'][idx_trial*setup.EXPERIMENT_SETUP['num_objects']:(idx_trial+1)*setup.EXPERIMENT_SETUP['num_objects']]
            prob_est = sesdata['est_values'][idx_trial*setup.EXPERIMENT_SETUP['num_objects']:(idx_trial+1)*setup.EXPERIMENT_SETUP['num_objects']]/100
            curr_subj_perf.append(np.nansum((pO.flatten()[stimOs]-prob_est)**2)) 
        
        perf_all.append(np.stack(curr_subj_perf))
        
        
    perf_all = np.stack(perf_all)
    m_perf = np.nanmean(perf_all, 0)
    sd_perf = np.nanstd(perf_all, 0)/np.sqrt(num_subj)

    plt.errorbar(np.arange(perf_all.shape[1])+1, m_perf, yerr=sd_perf, fmt='ko-', 
                 lw=5, markerfacecolor='white', markersize=15, markeredgewidth=5, 
                 capsize=5, elinewidth=5, capthick=5)

    # for est_trial_loc in setup.EXPERIMENT_SETUP['locEstimationTrials']:
    #     plt.arrow(est_trial_loc-window_size, 0.51, 0, 0.01, 
    #               head_width=1, head_length=0.01, color='k')

    plt.gca().spines.right.set_visible(False)
    plt.gca().spines.top.set_visible(False)
    # plt.ylim([0.52, 0.88])
    plt.xlabel('Value estimation bout', fontsize=25)
    plt.ylabel('Estimation error (SSE)', fontsize=25)
    plt.legend(fontsize=22, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_data_dir, "est_error_curve.pdf"))
    plt.close()

def fit_gt_value(data, figure_data_dir, use_mixed_effects):
    pO = setup.EXPERIMENT_SETUP['reward_schedule']
    pF = np.mean(setup.EXPERIMENT_SETUP['reward_schedule'], axis=1)

    num_est_trials = len(setup.EXPERIMENT_SETUP['locEstimationTrials'])

    all_coeffs = []
    all_ses = []
    all_ps = []

    all_Xs = []
    all_Ys = []
    for idx_trial in range(num_est_trials):
        for idx_subj, sesdata in enumerate(data):
            stimOs = sesdata['est_stims'][idx_trial*setup.EXPERIMENT_SETUP['num_objects']:(idx_trial+1)*setup.EXPERIMENT_SETUP['num_objects']]
            prob_est = sesdata['est_values'][idx_trial*setup.EXPERIMENT_SETUP['num_objects']:(idx_trial+1)*setup.EXPERIMENT_SETUP['num_objects']]/100
            stimF1 = stimOs//setup.EXPERIMENT_SETUP['num_vals']
            stimF2 = stimOs%setup.EXPERIMENT_SETUP['num_vals']
            
            subj_pO = pO.flatten()[stimOs]
            if sesdata['infDimension']==1:
                subj_pFinf = pF[stimF1]
                subj_pFnoninf = pF[stimF2]
            elif sesdata['infDimension']==2:
                subj_pFinf = pF[stimF2]
                subj_pFnoninf = pF[stimF1]
            else:
                raise ValueError

            all_Xs.append(np.stack([logit(subj_pFinf), logit(subj_pFnoninf), logit(subj_pO), 
                                    (sesdata['ID']*np.ones_like(stimOs)).astype(int), 
                                    (idx_trial*np.ones_like(stimOs)).astype(int)], 
                                    axis=1))
            all_Ys.append(logit(np.clip(prob_est, 1e-2, 1-1e-2)))
    
    all_Xs = np.concatenate(all_Xs, 0)
    all_Ys = np.concatenate(all_Ys, 0)[:, None]

    all_data = pd.DataFrame(np.concatenate([all_Xs, all_Ys], axis=1), columns=['pFinf', 'pFnoninf', 'pO', 'subj', 'bout', 'prob']).fillna(0)

    all_data.to_csv(os.path.join(processed_data_dir, f'fit_gt_value_df.csv'), index=False)

    mdl = smf.ols('prob~C(bout)*pFinf+C(bout)*pFnoninf+C(bout)*pO', all_data, missing='drop')
    mdlf = mdl.fit()
    all_coeffs.append(mdlf.params)
    all_ses.append(mdlf.bse)
    all_ps.append(mdlf.pvalues)
    print(mdlf.summary())
    print(mdlf.t_test([*[0]*4, 1, *[0]*3, -1, *[0]*7]))
            
    # if use_mixed_effects:
    #     all_coeffs = np.stack([coeffs.values for coeffs in all_coeffs])[:,1:]
    #     all_ses = np.stack([ses.values for ses in all_ses])[:,1:]
    #     all_ps = np.stack([ps.values[:4] for ps in all_ps])[:,1:]
    # else:
    #     all_coeffs = np.stack(all_coeffs)[:,1:]
    #     all_ses = np.stack(all_ses)[:,1:]
    #     all_ps = np.stack(all_ps)[:,1:]

    # var_names = ['F0', 'F1', 'O']

    # for i_var in range(0, all_coeffs.shape[1]):
    #     plt.errorbar(np.arange(1, num_est_trials+1), all_coeffs[:,i_var], all_ses[:,i_var], label=var_names[i_var], marker='o', capsize=5)
    # plt.xlabel('Value Estimation Trials')
    # plt.ylabel('Regression Weights')
    # plt.xticks(np.arange(1, num_est_trials+1))
    # plt.yticks(np.linspace(0, 2, 5))
    # plt.ylim([-0.2, 0.75])
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(figure_data_dir, "prob_est_betas.pdf"))
    # plt.close()
    return

def fit_anova(data, figure_data_dir, use_mixed_effects):
    num_est_trials = len(setup.EXPERIMENT_SETUP['locEstimationTrials'])

    all_eta2s = []

    all_Xs = []
    all_Ys = []

    for idx_trial in range(num_est_trials):
        for idx_subj, sesdata in enumerate(data):
            stimOs = sesdata['est_stims'][idx_trial*setup.EXPERIMENT_SETUP['num_objects']:(idx_trial+1)*setup.EXPERIMENT_SETUP['num_objects']]
            prob_est = sesdata['est_values'][idx_trial*setup.EXPERIMENT_SETUP['num_objects']:(idx_trial+1)*setup.EXPERIMENT_SETUP['num_objects']]/100
            stimF1 = stimOs//setup.EXPERIMENT_SETUP['num_vals']
            stimF2 = stimOs%setup.EXPERIMENT_SETUP['num_vals']
            
            if sesdata['infDimension']==1:
                subj_Finf = stimF1
                subj_Fnoninf = stimF2
            elif sesdata['infDimension']==2:
                subj_Finf = stimF2
                subj_Fnoninf = stimF1
            else:
                raise ValueError

            all_Xs.append(np.stack([subj_Finf, subj_Fnoninf, 
                                    (sesdata['ID']*np.ones_like(stimOs)).astype(int), 
                                    (idx_trial*np.ones_like(stimOs)).astype(int)], axis=1))
            all_Ys.append(logit(np.clip(prob_est, 1e-2, 1-1e-2)))
        
    all_Xs = np.concatenate(all_Xs, 0)
    all_Ys = np.concatenate(all_Ys, 0)[:, None]

    all_data = pd.DataFrame(np.concatenate([all_Xs, all_Ys], axis=1), columns=['Finf', 'Fnoninf', 'subj', 'bout', 'prob']).fillna(0)
    all_data.to_csv(os.path.join(processed_data_dir, f'fit_anova_df.csv'), index=False)

    mdl = smf.ols('prob~C(Finf)*C(Fnoninf)*C(bout)', all_data, missing='drop')
    mdlf = mdl.fit()
    # print(mdlf.summary())
    anova_tbl = sm.stats.anova_lm(mdlf, typ=3)
    all_eta2s.append(anova_tbl['sum_sq'][1:4]/(anova_tbl['sum_sq'][1:4]+anova_tbl['sum_sq'][4]))
    # all_ses.append(mdlf.bse)
    # all_ps.append(mdlf.pvalues)
    print(anova_tbl)
    
    all_eta2s = np.stack(all_eta2s)

    # var_names = ['F0', 'F1', 'O']

    # for i_var in range(all_eta2s.shape[1]):
    #     plt.plot(np.arange(1, num_est_trials+1), all_eta2s[:,i_var], label=var_names[i_var], marker='o')
    # plt.xlabel('Value Estimation Trials')
    # plt.ylabel('Partial Eta Squared')
    # plt.xticks(np.arange(1, num_est_trials+1))
    # plt.yticks(np.linspace(0, 0.08, 5))
    # plt.legend(loc='upper left')
    # plt.tight_layout()
    # plt.savefig(os.path.join(figure_data_dir, "prob_est_eta2.pdf"))
    # plt.close()
    return all_data

if __name__=='__main__':
    processed_data_dir = "data/Processed"
    processed_file_name = "processed"
    # processed_file_name = "excluded"
    figure_data_dir = "figures/All Processed"
    # figure_data_dir = "figures/All Excluded"
    with open(os.path.join(processed_data_dir, processed_file_name), "rb") as f:
        data = pickle.load(f)

    plot_value_error_curve(data, figure_data_dir)
    fit_gt_value_df = fit_gt_value(data, figure_data_dir, False)
    # fit_gt_value_df
    fit_anova_df = fit_anova(data, figure_data_dir, False)
    # fit_anova_df.to_csv(os.path.join(processed_data_dir, 'fit_anova_df.csv'), index=False)