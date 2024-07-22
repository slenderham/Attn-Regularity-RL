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

def manipulation_rt_to_params(data):
    num_subj = len(data)

    all_Xs = []
    all_Ys = []

    for idx_subj, sesdata in enumerate(data):
        manip_correct = sesdata['passive_question_correct']
        manip_rt = sesdata['passive_question_rt'].mean()
        
        
        all_Xs.append(np.stack([manip_correct, manip_rt], axis=1))