import numpy as np
import os, sys
from pathlib import Path

#############################
# 1. paths
#############################
# path to data
root_dir = Path('C:/data')
sys.path.append(os.path.dirname(os.getcwd()))
# Input/Output specs
idir = os.path.join(root_dir, 'ML1_data_input')
odir = os.path.join(root_dir, 'ML1_data_output')

#############################
# 2. run environment
#############################
# running on condor (default is false)
is_condor = True # BE CARE, OVERWRITTEN IN LOCAL CONSTANT
# numbers of cores to be used when multi-thread is possible, at least 4
nJobsForGridSearchCv = 4 # BE CARE, OVERWRITTEN IN LOCAL CONSTANT

#############################
# 3. General model parameters
#############################
# number of years of y data before the first year with features AND number of years for trend computation
ny_max_trend = 12

# Input data scaling. Admitted values:
# z_f: z-score features
# z_fl: z-score features and labels
# z_fl_au: z-score features and labels by AU
dataScaling = 'z_f'

# Add average y to feature set
# An alternative way to pass the admin-level unobserved effect to the model. The y average by AU is used as an
# additional feature. In the outer CV leave one year out loop, it uses the mean of training y,
# always scaled because features are scaled. It is alternative to DoOHEnc # (to be used only with DoOHEnc set to False).
# Admitted values: False, True
AddTargetMeanToFeature = False

# Scale One-Hot Encoding (normally set to False). Values: False, True
DoScaleOHEnc = False

# The cost function. Values: 'neg_root_mean_squared_error' and ‘r2’
scoringMetric = 'neg_root_mean_squared_error'

# number of groups (i.e. years) to be left out in the CV inner loop (for hyperpar setting)
n_out_inner_loop = 1

# Models are classified in
#      1. benchmarks (Null_model, PeakNDVI, Trend)
#      2. skModels (scikit-learn models).
# Options and feature definition do not apply to benchmarks.
# Benchmark model to be considered
benchmarks = ['Null_model', 'PeakNDVI', 'Trend']

# feature groups to be considered
feature_groups = {
    'rs_met': ['ND', 'NDmax', 'Rad', 'RainSum', 'T', 'Tmin', 'Tmax'],
    'rs_met_reduced': ['ND', 'RainSum', 'T'],
    'rs_met_sm_reduced': ['ND', 'RainSum', 'T', 'SM'], # test of ZA
    'rs': ['ND', 'NDmax'],
    'rs_reduced': ['ND'],
    'rs_sm_reduced': ['ND', 'SM'],
    'met': ['Rad', 'RainSum', 'T', 'Tmin', 'Tmax'],
    'met_reduced': ['Rad', 'RainSum', 'T'],
    'met_sm_reduced': ['Rad', 'RainSum', 'T','SM']
}
# dictionary for group labels used in plots
feature_groups2labels = {
    'rs_met':  'RS&Met',
    'rs_met_reduced': 'RS&Met-',
    'rs_met_sm_reduced': 'SM&RS&Met-',
    'rs': 'RS',
    'rs_reduced': 'RS-',
    'rs_sm_reduced': 'SM&RS-',
    'met': 'Met',
    'met_reduced': 'Met-',
    'met_sm_reduced': 'SM&Met-'
}

# percentage of features to be selected (as grid to be tested)
feature_prct_grid = [5, 10, 25, 50, 75, 100]

# Hyperparameters grid space
hyperparopt = 'grid'
if hyperparopt == 'grid':
    # Hidden layers treated as hyperparameters for NN MLP (1, 2 and 3 layers)
    # Set values that are exponents of two or values that can be divided by two
    hl2 = [(i, j) for i in [16, 32, 48, 64] for j in [16, 32, 48, 64]]
    hl3 = [[16, 32, 16], [16, 48, 16], [32, 48, 32], [32, 64, 32], [48, 64, 48], [32, 32, 32], [48, 48, 48], [64, 64, 64],
            [16, 16, 16]]
    hl = hl2 + hl3
    hyperGrid = dict(Lasso={'alpha': np.logspace(-5, 0, 13, endpoint=True)},
                     RandomForest={'max_depth': [10, 15, 20, 25, 30, 35, 40],  # Maximum number of levels in tree
                                   'max_features': ['auto', 'sqrt'],  # Number of features to consider at every split
                                   'n_estimators': [100, 250, 500],  # Number of trees in random forest
                                   'min_samples_split': np.linspace(0.2, 0.8, 6, endpoint=True)},
                     MLP={'alpha': np.logspace(-5, -1, 6, endpoint=True),
                          'hidden_layer_sizes': hl,
                          'activation': ['relu', 'tanh'],
                          'learning_rate': ['constant', 'adaptive']},
                     GBR={'learning_rate': [0.01, 0.05, 0.1],
                          # Empirical evidence suggests that small values of learning_rate favor better test error.
                          # [HTF] recommend to set the learning rate to a small constant (e.g. learning_rate <= 0.1)
                          # and choose n_estimators by early stopping.
                          'max_depth': [10, 20, 40],
                          'n_estimators': [100, 250, 500],
                          'min_samples_split': np.linspace(0.1, 0.8, 6, endpoint=True)},
                     SVR_linear={'gamma': np.logspace(-2, 2, 7, endpoint=True),
                                 # gamma defines how much influence a single training example has.
                                 # The larger gamma is, the closer other examples must be to be affected.
                                 'epsilon': np.logspace(-6, .5, 7, endpoint=True),
                                 'C': [1e-5, 1e-4, 1e-3, 1e-2, 1, 10, 100]},
                     SVR_rbf={'gamma': np.logspace(-2, 2, 7, endpoint=True),
                              'epsilon': np.logspace(-6, .5, 7, endpoint=True),
                              'C': [1e-5, 1e-4, 1e-3, 1e-2, 1, 10, 100]},#)#,
                     GPR1={'alpha': [1e-10, 1e-5, 1e-1]},
                     GPR2={'alpha': [1e-10, 1e-5, 1e-1, 0.05]})


#############################
# 4. Some housekeeping
#############################
# unnecessary columns to be dropped in stat file
drop_cols_stats = ['ASAP1_ID', 'AU_name', 'Region_ID']
# factor that divide production values to get production in desired units
production_scaler = 100000.0

##########################################
#    ENVIRONMENT SETTINGS
##########################################
# try importing any local settings if they exist
try:
    from .local_constants import *
except ImportError as e:
    pass


# EOF
