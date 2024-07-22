import numpy as np
import os
import pickle
from scipy.stats import binom, zscore
from scipy.signal import convolve
import pandas as pd
from matplotlib import pyplot as plt
import setup
import statsmodels.formula.api as smf
import itertools
import statsmodels.api as sm
plt.rc('font', size=15) 

def rt_regression(data, figure_data_dir):
    pO = setup.EXPERIMENT_SETUP['reward_schedule']
    pF = np.mean(setup.EXPERIMENT_SETUP['reward_schedule'], axis=1)

    all_Xs = []
    all_Ys = []

    # log(RT) ~ trial * ((log_odds_0 + log_odds_1 + log_odds_obj) +
    #           rwd_prev * (F0_chosen_repeated + F1_chosen_repeated + obj_chosen_repeated + 
    #                       F0_unchosen_repeated + F1_unchosen_repeated + obj_unchosen_repeated))

    for idx_subj, sesdata in enumerate(data):
        # get stimuli options, choice, and rewards for both options
        stimOs = sesdata['stim']
        choices = sesdata['choice']
        rwd_full = sesdata['full_reward']
        rts = sesdata['reaction_times']
        skipped_mask = np.logical_not(sesdata['skipped'][:-1]) & np.logical_not(sesdata['skipped'][1:]) # skip trials whose previous trial was skipped

        stimF1 = stimOs//setup.EXPERIMENT_SETUP['num_vals']
        stimF2 = stimOs%setup.EXPERIMENT_SETUP['num_vals']
        
        # get reward probabilities along two dimensions
        subj_pO = pO.flatten()[stimOs[1:]] # discard first trial
        if sesdata['infDimension']==1:
            subj_pFinf = pF[stimF1[1:]]
            subj_pFnoninf = pF[stimF2[1:]]
        else:
            subj_pFinf = pF[stimF2[1:]]
            subj_pFnoninf = pF[stimF1[1:]]

        # get the features
        if sesdata['infDimension']==1:
            stimFinf = stimF1
            stimFnoninf = stimF2
        else:
            stimFinf = stimF2
            stimFnoninf = stimF1
        
        # include the log odds of reward for both options
        # TODO: should I separate these? necessary?

        subj_Xs_objective_values = np.stack([np.log(subj_pFinf[skipped_mask,1])-np.log(subj_pFinf[skipped_mask,0]), \
                                             np.log(subj_pFnoninf[skipped_mask,1])-np.log(subj_pFnoninf[skipped_mask,0]), \
                                             np.log(subj_pO[skipped_mask,1])-np.log(subj_pO[skipped_mask,0])], axis=1)

        def make_X_prev_rew(rwd_pre, c, skipped_mask):
            rwd_pre = rwd_pre[:-1][skipped_mask]
            c = c[:-1][skipped_mask]
            r_pre_chosen = rwd_pre[np.arange(np.sum(skipped_mask)), c] # rwd from chosen option
            r_pre_unchosen = rwd_pre[np.arange(np.sum(skipped_mask)), 1-c] # rwd from unchosen option
            return np.stack([r_pre_chosen, r_pre_unchosen], axis=1)

        def make_X_prev_ch(stims, c, skipped_mask):
            c = c[:-1][skipped_mask]
            stim_pre = stims[:-1][skipped_mask]
            stim_post = stims[1:][skipped_mask]
            pre_chosen = stim_pre[np.arange(np.sum(skipped_mask)), c] # previously chosen stim, N
            pre_unchosen = stim_pre[np.arange(np.sum(skipped_mask)), 1-c] # previously unchosen stim, N

            chosen_stim_ch = (pre_chosen[:,None]==stim_post).astype(float).sum(-1) # calculate if the previous stimuli and late stimuli share the same feature, N 
            unchosen_stim_ch = (pre_unchosen[:,None]==stim_post).astype(float).sum(-1) # calculate if the previous stimuli and late stimuli share the same feature, N
            return np.stack([chosen_stim_ch, unchosen_stim_ch], axis=1)

        subj_Xs_credit_assignment =  np.concatenate([
            make_X_prev_rew(rwd_full, choices, skipped_mask),
            make_X_prev_ch(stimFinf, choices, skipped_mask), \
            make_X_prev_ch(stimFnoninf, choices, skipped_mask), \
            make_X_prev_ch(stimOs, choices, skipped_mask)], axis=1)
        
        all_Xs.append(np.concatenate([
            np.arange(1,setup.EXPERIMENT_SETUP['numChoiceTrials'])[skipped_mask][:,None],
            subj_Xs_objective_values,
            subj_Xs_credit_assignment,
            (sesdata['ID']*np.ones((np.sum(skipped_mask), 1))).astype(int)
        ], axis=1))
         
        all_Ys.append(rts[1:][skipped_mask]/1000)

    all_Xs = np.concatenate(all_Xs, 0)
    all_Ys = np.concatenate(all_Ys, 0)[:, None]
    all_data = pd.DataFrame(np.concatenate([all_Xs, all_Ys], axis=1), columns=['trial', 'pFinf', 'pFnoninf', 'pO', 
                                                                               'past_rew_ch', 'past_rew_unch',
                                                                               'past_Finf_ch', 'past_Finf_unch', 
                                                                               'past_Fnoninf_ch', 'past_Fnoninf_unch',
                                                                               'past_O_ch', 'past_O_unch',
                                                                                'subj', 'rts'])
    all_data.to_csv(os.path.join(processed_data_dir, f'reaction_time_analysis_df.csv'), index=False)

    mdl = smf.glm('rts~trial*(pFinf+pFnoninf+pO+\
                    past_rew_ch*(past_Finf_ch+past_Fnoninf_ch+past_O_ch)+\
                    past_rew_unch*(past_Finf_unch+past_Fnoninf_unch+past_O_unch))', data=all_data,
                    family=sm.families.Gamma(sm.genmod.families.links.Log()))
    mdlf = mdl.fit()

    print(mdlf.summary())

if __name__=='__main__':
    processed_data_dir = "data/Processed"
    processed_file_name = "processed"
    # processed_file_name = "excluded"
    figure_data_dir = "figures/All Processed"
    # figure_data_dir = "figures/All Excluded"
    with open(os.path.join(processed_data_dir, processed_file_name), "rb") as f:
        data = pickle.load(f)

    rt_regression(data, figure_data_dir)