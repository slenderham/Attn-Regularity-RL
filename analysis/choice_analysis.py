import numpy as np
import os
import pickle
from scipy.stats import binom
from scipy.signal import convolve
import pandas as pd
from matplotlib import pyplot as plt
import setup
import statsmodels.formula.api as smf
import itertools
import statsmodels.api as sm
plt.rc('font', size=25, family='arial') 
plt.rc('axes', linewidth=2.5)
plt.rc('xtick.major', width=2, size=8)
plt.rc('ytick.major', width=2, size=8)

def nanmovmean(x, window_size):
    x_len = len(x)
    smth_x = []
    for t in range(x_len-window_size):
        smth_x.append(np.nanmean(x[t:t+window_size]))
    return np.stack(smth_x)

def plot_learning_curve(data, figure_data_dir):
    num_trials = data[0]['numTrials']
    num_subj = len(data)
    perf_all = []
    rwd_all = []
    window_size = 20
    for idx_subj, sesdata in enumerate(data):
        unambiguous_choose_better = sesdata['choose_better'].astype(float)
        unambiguous_choose_better[sesdata['probs'][:,0]==sesdata['probs'][:,1]] = np.nan
        perf_all.append(nanmovmean(unambiguous_choose_better, window_size))
        rwd_all.append(nanmovmean(sesdata['reward'], window_size))
        # plt.plot(perf_all[-1].T)
        # plt.show()
    perf_all = np.stack(perf_all)
    rwd_all = np.stack(rwd_all)

    m_perf = np.nanmean(perf_all, 0)
    m_rwd = np.nanmean(rwd_all, 0)

    sd_perf = np.nanstd(perf_all, 0)/np.sqrt(num_subj)
    sd_rwd = np.nanstd(rwd_all, 0)/np.sqrt(num_subj)

    plt.plot(m_perf, lw=5, color='grey', label='Prob. correct')
    plt.fill_between(np.arange(perf_all.shape[1]), m_perf-sd_perf, m_perf+sd_perf, color=(0.5, 0.5, 0.5, 0.3))
    plt.plot(m_rwd, lw=5, color='black', label='Reward')
    plt.fill_between(np.arange(rwd_all.shape[1]), m_rwd-sd_rwd, m_rwd+sd_rwd, color=(0, 0, 0, 0.3))

    # for est_trial_loc in setup.EXPERIMENT_SETUP['locEstimationTrials']:
    #     plt.arrow(est_trial_loc-window_size, 0.51, 0, 0.01, 
    #               head_width=1, head_length=0.01, color='k')

    plt.gca().spines.right.set_visible(False)
    plt.gca().spines.top.set_visible(False)
    plt.ylim([0.52, 0.88])
    plt.xlabel('Trial number', fontsize=25)
    plt.ylabel('Performance', fontsize=25)
    plt.legend(fontsize=22, frameon=False, labelspacing = 0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_data_dir, "learning_curve.pdf"))
    plt.close()

def credit_assignment(data, figure_data_dir, use_mixed_effects):
    # find chosen feedback, unchosen feedback
    num_subj = len(data)
    
    total_num_blocks = 2

    all_Xs = []
    all_Ys = []

    for num_block in range(total_num_blocks):
        num_trials_per_block = setup.EXPERIMENT_SETUP['numChoiceTrials']//total_num_blocks
        num_trials_to_fit = np.arange(num_block*num_trials_per_block, \
                                      min((num_block+1)*num_trials_per_block, setup.EXPERIMENT_SETUP['numChoiceTrials']-1))

        for idx_subj, sesdata in enumerate(data):
            stims_pre = sesdata['stim'][num_trials_to_fit]
            rwd_full_pre = sesdata['full_reward'][num_trials_to_fit]
            stims_post = sesdata['stim'][num_trials_to_fit+1]
            skipped_mask = np.logical_not(sesdata['skipped'][num_trials_to_fit]) & np.logical_not(sesdata['skipped'][num_trials_to_fit+1])

            stimF1_pre = stims_pre//setup.EXPERIMENT_SETUP['num_vals']
            stimF2_pre = stims_pre%setup.EXPERIMENT_SETUP['num_vals']

            stimF1_post = stims_post//setup.EXPERIMENT_SETUP['num_vals']
            stimF2_post = stims_post%setup.EXPERIMENT_SETUP['num_vals']

            if sesdata['infDimension']==1:
                stimFinf_pre = stimF1_pre
                stimFnoninf_pre = stimF2_pre
                stimFinf_post = stimF1_post
                stimFnoninf_post = stimF2_post
            else:
                stimFinf_pre = stimF2_pre
                stimFnoninf_pre = stimF1_pre
                stimFinf_post = stimF2_post
                stimFnoninf_post = stimF1_post

            choices = sesdata['choice'][num_trials_to_fit]
            target = sesdata['choice'][num_trials_to_fit+1]

            def make_X(pre, rwd_pre, post, c):
                pre_chosen = pre[np.arange(np.sum(skipped_mask)), c]
                pre_unchosen = pre[np.arange(np.sum(skipped_mask)), 1-c]

                r_pre_chosen = rwd_pre[np.arange(np.sum(skipped_mask)), c]
                r_pre_unchosen = rwd_pre[np.arange(np.sum(skipped_mask)), 1-c]

                chosen_stim_ch = (pre_chosen[:,None]==post).astype(float) # calculate if the previous stimuli and late stimuli share the same feature, N x 2
                unchosen_stim_ch = (pre_unchosen[:,None]==post).astype(float) # calculate if the previous stimuli and late stimuli share the same feature, N x 2

                chosen_stim_rw = chosen_stim_ch*(2*r_pre_chosen[:,None]-1) # flip the sign if the previous chosen stim was not rewarded
                unchosen_stim_rw = unchosen_stim_ch*(2*r_pre_unchosen[:,None]-1)

                chosen_stim_ch = chosen_stim_ch[:,1]-chosen_stim_ch[:,0] # +1 if right option shared, -1 if left option share, 0 if neither
                unchosen_stim_ch = unchosen_stim_ch[:,1]-unchosen_stim_ch[:,0] # +1 if right option shared, -1 if left option share, 0 if neither

                chosen_stim_rw = chosen_stim_rw[:,1]-chosen_stim_rw[:,0]
                unchosen_stim_rw = unchosen_stim_rw[:,1]-unchosen_stim_rw[:,0]

                return np.stack([chosen_stim_rw, chosen_stim_ch, unchosen_stim_rw, unchosen_stim_ch], axis=1)
                
            # predictors are inf dim R chosen, inf dim C chosen, noninf dim R chosen, noninf dim C chosen, 
            subj_Xs = np.concatenate([
                make_X(stimFinf_pre[skipped_mask], rwd_full_pre[skipped_mask], stimFinf_post[skipped_mask], choices[skipped_mask]), \
                make_X(stimFnoninf_pre[skipped_mask], rwd_full_pre[skipped_mask], stimFnoninf_post[skipped_mask], choices[skipped_mask]), \
                make_X(stims_pre[skipped_mask], rwd_full_pre[skipped_mask], stims_post[skipped_mask], choices[skipped_mask]), \
                (sesdata['ID']*np.ones((np.sum(skipped_mask), 1))).astype(int), \
                (num_block*np.ones((np.sum(skipped_mask), 1))), \
                    ], axis=1)
            
            all_Xs.append(subj_Xs)
            all_Ys.append(target[skipped_mask])

            # plt.imshow((subj_Xs[:,:-1]), cmap='seismic', vmin=-2, vmax=2)
            # plt.colorbar()
            # plt.show()

    all_Xs = np.concatenate(all_Xs, axis=0)
    all_Ys = np.concatenate(all_Ys, axis=0)[:,None]

    var_names = ['F0', 'F1', 'O']
    rw_ch = ['R', 'C']
    ch_unch = ['Chosen', 'Foregone']

    all_var_names = itertools.product(var_names, ch_unch, rw_ch)
    all_var_names = ['_'.join(s) for s in all_var_names]

    all_data = pd.DataFrame(np.concatenate([all_Xs, all_Ys], axis=1), columns=[*all_var_names, 'subj', 'block', 'choice'])
    all_data.to_csv(os.path.join(processed_data_dir, f'credit_assignment_df.csv'), index=False)

    mdl = smf.glm('choice~block*('+'+'.join(all_var_names)+')', data=all_data, family=sm.families.Binomial())
    mdlf = mdl.fit()
    print(mdlf.summary())
    all_coeffs = mdlf.params[1:]
    all_ses = mdlf.bse[1:]
    all_ps = mdlf.pvalues[1:]

    # all_xlabels = itertools.product(ch_unch, rw_ch)
    # all_xlabels = ['_'.join(s) for s in all_xlabels]

    # delta_x = [-0.2, 0, 0.2]
    # for i in range(len(var_names)):
    #     plt.bar(np.arange(1,len(rw_ch)*len(ch_unch)+1)+delta_x[i], \
    #             all_coeffs[i*len(rw_ch)*len(ch_unch):(i+1)*len(rw_ch)*len(ch_unch)], 0.2, \
    #             yerr=all_ses[i*len(rw_ch)*len(ch_unch):(i+1)*len(rw_ch)*len(ch_unch)], label=var_names[i], capsize=5)
    # plt.ylabel('Regression Weights')
    # plt.ylim([-0.7, 0.7])
    # plt.xticks(np.arange(1, (len(rw_ch)*len(ch_unch))+1), labels=all_xlabels)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(figure_data_dir, f"credit_assignment_{num_block}.pdf"))
    # plt.close()
    return

def steady_state_choice_analysis(data, figure_data_dir, use_mixed_effects):
    pO = setup.EXPERIMENT_SETUP['reward_schedule']
    pF = np.mean(setup.EXPERIMENT_SETUP['reward_schedule'], axis=1)

    all_coeffs = []
    all_ses = []
    all_ps = []

    total_num_blocks = 2
    
    all_Xs = []
    all_Ys = []

    for num_block in range(total_num_blocks):
        num_trials_per_block = setup.EXPERIMENT_SETUP['numChoiceTrials']//total_num_blocks
        num_trials_to_fit = np.arange(num_block*num_trials_per_block, (num_block+1)*num_trials_per_block)

        for idx_subj, sesdata in enumerate(data):
            stimOs = sesdata['stim'][num_trials_to_fit]
            choices = sesdata['choice'][num_trials_to_fit]
            skipped_mask = np.logical_not(sesdata['skipped'][num_trials_to_fit])
            
            stimF1 = stimOs//setup.EXPERIMENT_SETUP['num_vals']
            stimF2 = stimOs%setup.EXPERIMENT_SETUP['num_vals']
            
            subj_pO = pO.flatten()[stimOs]
            if sesdata['infDimension']==1:
                subj_pFinf = pF[stimF1]
                subj_pFnoninf = pF[stimF2]
            else:
                subj_pFinf = pF[stimF2]
                subj_pFnoninf = pF[stimF1]
            
            all_Xs.append(np.stack([np.log(subj_pFinf[skipped_mask,1]/subj_pFinf[skipped_mask,0]), \
                                    np.log(subj_pFnoninf[skipped_mask,1]/subj_pFnoninf[skipped_mask,0]), \
                                    np.log(subj_pO[skipped_mask,1]/subj_pO[skipped_mask,0]), \
                                    (sesdata['ID']*np.ones(np.sum(skipped_mask))).astype(int), \
                                    2*(num_block*np.ones(np.sum(skipped_mask)))-1], axis=1))
            all_Ys.append(choices[skipped_mask])
        
    all_Xs = np.concatenate(all_Xs, 0)
    all_Ys = np.concatenate(all_Ys, 0)[:, None]
    all_data = pd.DataFrame(np.concatenate([all_Xs, all_Ys], axis=1), 
                            columns=['pFinf', 'pFnoninf', 'pO', 'subj', 'block', 'prob']).fillna(0)
    all_data.to_csv(os.path.join(processed_data_dir, f'choice_curve_df.csv'), index=False)
    # all_data = all_data[all_data['block']>0]
    mdl = smf.glm('prob~(pFinf+pFnoninf+pO)', all_data, missing='drop', family=sm.families.Binomial())
    mdlf = mdl.fit()
    print(mdlf.summary())
    print(mdlf.t_test([0, 1, -1, 0]))
    all_coeffs.append(mdlf.params[1:])
    all_ses.append(mdlf.bse[1:])
    all_ps.append(mdlf.pvalues[1:])

   
    # var_names = ['F0', 'F1', 'O']
    # for i in range(len(var_names)):
    #     plt.errorbar(np.arange(1, total_num_blocks+1), all_coeffs[:,i], all_ses[:,i], capsize=5, label=var_names[i])
    # # plt.bar(np.arange(1, len(var_names)+1), all_coeffs, color='grey')
    # # plt.errorbar(np.arange(1, len(var_names)+1), all_coeffs, all_ses, , linestyle="", color='k')
    # plt.xlabel('Choice Trial Block')
    # plt.ylabel('Regression Weights')
    # # plt.yticks(np.linspace(0, 2, ))
    # plt.xlim([0.75, total_num_blocks+0.25])
    # plt.xticks(np.arange(1, total_num_blocks+1))
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(figure_data_dir, "choice_curves_slope.pdf"))
    # plt.close()
    return

if __name__=='__main__':
    processed_data_dir = "data/Processed"
    processed_file_name = "processed"
    # processed_file_name = "excluded"
    figure_data_dir = "figures/All Processed"
    # figure_data_dir = "figures/All Excluded"
    with open(os.path.join(processed_data_dir, processed_file_name), "rb") as f:
        data = pickle.load(f)

    plot_learning_curve(data, figure_data_dir)
    steady_state_choice_analysis(data, figure_data_dir, False)
    credit_assignment(data, figure_data_dir, False)