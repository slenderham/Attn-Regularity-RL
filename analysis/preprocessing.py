import pandas as pd
import json as json
import glob
import os
import re
import pickle
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import setup

def read_subj(filename, subj_ID):
    df = pd.read_csv(filename, header=1)
    sesdata = {}

    # read metadata (informative dimension)
    sesdata['infDimension'] = int(df['infDimension'][1])
    sesdata['ID'] = int(subj_ID)

    # read choice data
    choice_df = df[['reactionTime','chooseLeft',\
                    'stimulusLeft','stimulusRight',\
                    'rewardLeft','rewardRight',\
                    'probLeft','probRight', 'skipped', \
                    'reactionTime']][df['typeOfTrial']=='choice']
    sesdata['numTrials'] = choice_df.shape[0]

    sesdata['skipped'] = choice_df['skipped'].to_numpy().astype(bool)
    sesdata['reaction_times'] = choice_df['reactionTime'].to_numpy() # get reaction times, N
    sesdata['full_reward'] = choice_df[['rewardLeft','rewardRight']].to_numpy() # get reward outcomes, Nx2
    sesdata['probs'] = choice_df[['probLeft','probRight']].to_numpy() # get probabilities of reward, Nx2
    sesdata['choice'] = choice_df['chooseLeft'].fillna(False).to_numpy() # choice, N
    sesdata['choice'] = 1-sesdata['choice'].astype(int) 
    sesdata['choice'][sesdata['skipped']] = -1
    sesdata['choose_better'] = (1-sesdata['choice'])*(sesdata['probs'][:,0]>=sesdata['probs'][:,1]) \
                             + (sesdata['choice'])*(sesdata['probs'][:,0]<=sesdata['probs'][:,1]) 
    sesdata['choose_better'][sesdata['skipped']] = 0
    sesdata['reward'] = np.zeros(sesdata['numTrials'])
    sesdata['reward'][np.logical_not(sesdata['skipped'])] = \
        sesdata['full_reward'][np.arange(sesdata['numTrials'])[np.logical_not(sesdata['skipped'])], \
                               sesdata['choice'][np.logical_not(sesdata['skipped'])]] # reward obtained, N
    sesdata['reaction_times'] = choice_df['reactionTime'].to_numpy()[:,0]

    print(f"Manipulated Dim: {sesdata['infDimension']}, Total reward: {np.nansum(sesdata['reward'])}, Proportion Better: {np.mean(sesdata['choose_better']).round(2)}")
    num_unambiguous_trials = sesdata['probs'][setup.ANALYSIS_SETUP['performance_test_after_trial']:,0]\
                            !=sesdata['probs'][setup.ANALYSIS_SETUP['performance_test_after_trial']:,1] # unambiguous trials where both options are not equally rewarding
    
    
    def read_stim_from_string_choice(row):
        left_stim = json.loads(row['stimulusLeft'])
        right_stim = json.loads(row['stimulusRight'])
        return np.array([(left_stim['infDim1']-1)*4+left_stim['infDim2']-1, \
                         (right_stim['infDim1']-1)*4+right_stim['infDim2']-1])

    sesdata['stim'] = np.stack(choice_df.apply(read_stim_from_string_choice, axis=1).to_list()) # stimulus object

    # read estimation data
    est_df = df[['estimationStimulus', 'estimationValue']][df['typeOfTrial']=='estimation']

    def read_stim_from_string_est(row):
        stim = json.loads(row['estimationStimulus'])
        return (stim['infDim1']-1)*4+stim['infDim2']-1

    sesdata['est_stims'] = est_df.apply(read_stim_from_string_est, axis=1).to_numpy()
    sesdata['est_values'] = est_df['estimationValue'].to_numpy()

    passive_df = df[['correct', 'reactionTime']][df['typeOfTrial']=='question']
    sesdata['passive_question_correct'] = passive_df['correct'].to_numpy()
    sesdata['passive_question_rt'] = passive_df['reactionTime'].to_numpy()

    flat_rwd_schedule = setup.EXPERIMENT_SETUP['reward_schedule'].reshape((-1))
    nan_value_ests = np.isnan(sesdata['est_values'][setup.ANALYSIS_SETUP['value_est_accuracy_after_trial']:])
    val_est_acc = stats.pearsonr(
        flat_rwd_schedule[sesdata['est_stims'][setup.ANALYSIS_SETUP['value_est_accuracy_after_trial']:]][~nan_value_ests], \
        sesdata['est_values'][setup.ANALYSIS_SETUP['value_est_accuracy_after_trial']:][~nan_value_ests])

    print(np.sum(sesdata['skipped'])+np.sum(np.isnan(sesdata['est_values'])))
    if stats.binomtest(np.sum(sesdata['choose_better'][setup.ANALYSIS_SETUP['performance_test_after_trial']:][num_unambiguous_trials]), 
                       np.sum(num_unambiguous_trials), 0.5, alternative='greater').pvalue >= setup.ANALYSIS_SETUP['performance_test_threshold']\
       and ((val_est_acc.correlation <= 0) or (val_est_acc.pvalue >= setup.ANALYSIS_SETUP['value_est_accuracy_threshold'])):
        print(np.sum(sesdata['choose_better'][setup.ANALYSIS_SETUP['performance_test_after_trial']:][num_unambiguous_trials])/np.sum(num_unambiguous_trials))
        print(stats.pearsonr(flat_rwd_schedule[sesdata['est_stims'][setup.ANALYSIS_SETUP['value_est_accuracy_after_trial']:]][~nan_value_ests], \
                              sesdata['est_values'][setup.ANALYSIS_SETUP['value_est_accuracy_after_trial']:][~nan_value_ests]))
        print('failed to pass both performance threshold and value estimation criterion')
        print(setup.MISC['print_section_divider'])
        return False, sesdata
    print(np.sum(sesdata['choose_better'][setup.ANALYSIS_SETUP['performance_test_after_trial']:][num_unambiguous_trials])/np.sum(num_unambiguous_trials))
    print(stats.pearsonr(flat_rwd_schedule[sesdata['est_stims'][setup.ANALYSIS_SETUP['value_est_accuracy_after_trial']:]][~nan_value_ests], \
                            sesdata['est_values'][setup.ANALYSIS_SETUP['value_est_accuracy_after_trial']:][~nan_value_ests]))

    if stats.binomtest(np.sum(sesdata['passive_question_correct']), len(sesdata['passive_question_correct']), 0.5, alternative="greater").pvalue \
        >= setup.ANALYSIS_SETUP['passive_question_threshold']:
        print(np.sum(sesdata['passive_question_correct'])/len(sesdata['passive_question_correct']))
        print('failed to pass passive trial criterion')
        print(setup.MISC['print_section_divider'])
        return False, sesdata

    print(setup.MISC['print_section_divider'])

    return True, sesdata

def read_all_subj():
    data_dirs = ["/Users/f005d7d/Documents/Attn_MdPRL/Py-attention-project-analysis/data/attention experiment data Fall 2022",
                 "/Users/f005d7d/Documents/Attn_MdPRL/Py-attention-project-analysis/data/attention experiment data Winter 2023",
                 "/Users/f005d7d/Documents/Attn_MdPRL/Py-attention-project-analysis/data/attention experiment data Spring 2023"]
    # data_dirs = ["/Users/f005d7d/Documents/Attn_MdPRL/Py-attention-project-analysis/data/sunapee high"]
    all_sesdata = []
    all_sesdata_to_exclude = []
    tot_subj_num = 0
    all_subj_acc_rej_status = []
    print(setup.MISC['print_section_divider'])
    for data_dir in data_dirs:
        file_path_pttrn = os.path.join(data_dir, "DEBUG_worker_*.csv")
        for _, subj_file in enumerate(glob.glob(file_path_pttrn)):
            tot_subj_num += 1
            subj_ID = re.search("DEBUG_worker_(.+?).csv", subj_file).group(1)
            print(f"Processing subject {subj_ID}")
            passed, subj_data = read_subj(subj_file, subj_ID)
            all_subj_acc_rej_status.append([subj_ID, passed])
            if passed:
                all_sesdata.append(subj_data)
            else:
                all_sesdata_to_exclude.append(subj_data)
    print(f"{len(all_sesdata)}/{tot_subj_num} subjects passed the performance threshold")
    all_subj_acc_rej_status = pd.DataFrame(all_subj_acc_rej_status, columns=["ID", "Accepted"])
    all_subj_acc_rej_status.to_csv(os.path.join(processed_data_dir,"subject_accept_reject_status.csv"))
    return all_sesdata, all_sesdata_to_exclude

if __name__=='__main__':
    processed_data_dir = "data/Processed"
    processed_file_name = "processed"
    excluded_file_name = "excluded"
    passed_data, excluded_data = read_all_subj()
    with open(os.path.join(processed_data_dir, processed_file_name), 'wb') as f:
        pickle.dump(passed_data, f)
    with open(os.path.join(processed_data_dir, excluded_file_name), 'wb') as f:
        pickle.dump(excluded_data, f)