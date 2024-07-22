import numpy as np

EXPERIMENT_SETUP = {
    'reward_schedule': np.array([
        [0.7, 0.9, 0.1, 0.9],
        [0.9, 0.3, 0.9, 0.7],
        [0.1, 0.9, 0.1, 0.1],
        [0.9, 0.7, 0.1, 0.1]
    ]),
    'num_objects': 16,
    'num_dims': 2,
    'num_vals': 4,
    'numChoiceTrials': 128,
    'numBlocks': 16,
    'numChoiceTrialsPerBlock': 8,
    'locEstimationTrials': np.arange(31, 128, 32)
}

ANALYSIS_SETUP = {
    'performance_test_after_trial': 64,
    'performance_test_threshold': 0.05,
    'value_est_accuracy_after_trial': 32,
    'value_est_accuracy_threshold': 0.05,
    'passive_question_threshold': 0.05,
}

MISC = {
    'print_section_divider': "="*65
}