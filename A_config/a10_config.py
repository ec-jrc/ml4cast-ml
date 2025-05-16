#import pandas as pd
import sys

import numpy as np
import os
import json
import sys
from datetime import datetime

class read_ope:
  def __init__(self, full_path_config):
    # Get the path
    path = os.path.dirname(full_path_config)
    # Get the base name and extension
    base_name = os.path.basename(full_path_config)
    extension = os.path.splitext(base_name)[1]
    base_name_no_ext = os.path.splitext(base_name)[0]
    full_path_config = os.path.join(path, base_name_no_ext + '_ope' + extension)
    with open(full_path_config, 'r') as fp:
        jdict = json.load(fp)
    # month X means that all months up to X (included) are used, so this is possible in month X+1
    self.forecastingMonth = int(jdict['forecastingMonth'])
    self.Year = int(jdict['Year'])

class read:
  def __init__(self, full_path_config, run_name, run_type=None):

    with open(full_path_config, 'r') as fp:
        jdict = json.load(fp)
    self.AOI = jdict['AOI']
    # year start and end define the period for which I have yield data and RS data (used for tuning)
    self.year_start = int(jdict['year_start'])
    self.year_end = int(jdict['year_end'])
    # Percentage of admin to retain (based on area). I.e. rank by area and retain only the largest %
    self.prct2retain = int(jdict['prct2retain'])
    #self.forecastingMonths = jdict['forecastingMonths'] obsolete, now %
    self.crops = jdict['crops']
    self.afi = jdict['afi']
    if "crop_au_exclusions" in jdict:
        self.crop_au_exclusions = jdict['crop_au_exclusions']
    else:
        self.crop_au_exclusions = {}

    self.root_dir = jdict['root_dir']
    self.data_dir = os.path.join(self.root_dir, jdict['data_dir'])
    self.output_dir = os.path.join(self.root_dir, 'RUN_' + self.afi)
    self.fn_reference_shape = jdict['fn_reference_shape']
    self.country_name_in_shp_file = jdict['country_name_in_shp_file']
    self.gaul0_column_name_in_shp_file = jdict['gaul0_column_name_in_shp_file']
    self.adminID_column_name_in_shp_file = jdict['adminID_column_name_in_shp_file']

    self.ope_data_dir = os.path.join(self.root_dir, jdict['ope_data_dir'])
    run_stamp = datetime.today().strftime('%Y%m%d')
    self.ope_run_dir = os.path.join(self.output_dir, 'OPE_'+ run_name + '_made_' + str(run_stamp))
    self.ope_run_out_dir = os.path.join(self.ope_run_dir, 'output')
    self.models_dir = os.path.join(self.output_dir, 'TUNE_' + run_name)
    self.models_spec_dir = os.path.join(self.models_dir, 'Specs')
    if run_type == 'fast_tuning':
        self.models_out_dir = os.path.join(self.models_dir, 'Output_fast_tuning')
    else:
        self.models_out_dir = os.path.join(self.models_dir, 'Output')

    self.ivars = jdict['ivars']
    self.ivars_short = jdict['ivars_short']
    self.ivars_units = jdict['ivars_units']
    self.sos = int(jdict['sos'])
    self.eos = int(jdict['eos'])
    # take all the month where SOS fall in
    self.sosMonth = int(np.ceil(self.sos / 3)) # note ceil makes it correct, think about it
    # take all the month where EOS fall in
    self.eosMonth = int(np.ceil(self.eos / 3))  # same here
    self.yield_units = jdict['yield_units']
    self.area_unit =jdict['area_unit']
    # factor that divide production values to get production in desired units
    # self.production_scaler =jdict['production_scaler']
    # get forecasting months from season%
    self.forecastingPrct = jdict['forecastingPrct']
    if self.sosMonth < self.eosMonth:
        real_months = list(range(int(self.sosMonth), int(self.eosMonth + 1)))
    else:
        real_months = list(range(int(self.sosMonth), 12 + 1)) + list(range(1, int(self.eosMonth) + 1))

    id_months = np.array(range(1,len(real_months)+1))
    prct_months = id_months/len(id_months)*100
    self.forecastingMonths = []
    self.forecastingCalendarMonths = []
    for prct in self.forecastingPrct:
        self.forecastingMonths = self.forecastingMonths + list([int(id_months[np.argmin(np.abs(prct_months-float(prct)))])])
        self.forecastingCalendarMonths =self.forecastingCalendarMonths + list([int(real_months[np.argmin(np.abs(prct_months-float(prct)))])])

class mlSettings:
  def __init__(self, forecastingMonths=0):
    # Define settings used in the ML workflow

    #set forcasting month (1 is the first)
    self.forecastingMonths = forecastingMonths

    # scikit, numbers of cores to be used when multi-thread is possible, at least 4
    # self.nJobsForGridSearchCv = 4 # attempt to solve overuse of cpu in condor
    if 'win' in sys.platform:
        self.nJobsForGridSearchCv = 8
    else:
        # Three params are needed for condor:
        # a) nJobsForGridSearchCv
        # b) NUM_THREADS (used in condor launcher)
        # c) request_cpus (used in condor.submit_template)
        # c must be a*b or I will use more CPU than requested. Requesting more cpu make job allocation slower
        self.condor_param = {'nJobsForGridSearchCv': 1, 'NUM_THREADS': 1, 'request_cpus': 1}
        #self.nJobsForGridSearchCv = 1
        self.nJobsForGridSearchCv = self.condor_param['nJobsForGridSearchCv']

    # Input data scaling. Admitted values:
    # z_f: z-score features
    # z_fl: z-score features and labels
    # z_fl_au: z-score features and labels by AU
    self.dataScaling = 'z_f'

    # The cost function. Values: 'neg_root_mean_squared_error' and ‘r2’
    self.scoringMetric = 'neg_root_mean_squared_error'

    # Models are classified in
    #      1. benchmarks (Null_model, PeakNDVI, Trend)
    #      2. skModels (scikit-learn models).
    # Options and feature definition do not apply to benchmarks.
    # Benchmark model to be considered
    self.benchmarks = ['Null_model', 'PeakNDVI', 'Trend']
    # Feature engineering types (all ML model will be tested using default monthly values and these ft eng settings)
    # ft settings must start with @
    self.ft_eng = ['@PeakFPARAndLast3'] # @PeakFPARAndLast3 change features groups in c100 (changes rs and does not apply to met)

    # feature groups to be considered
    rad_var = 'rad' #sometimes is 'Rad'
    bio_var = 'FP' # could be FP or ND ... se config ivars and ivars short
    self.rad_var = rad_var
    self.bio_var = bio_var
    self.feature_groups = {
      'rs_met': [bio_var, bio_var + 'max', rad_var, 'RainSum', 'T', 'Tmin', 'Tmax'],
      'rs_met_reduced': [bio_var, 'RainSum', 'T'],
      'rs_met_sm_reduced': [bio_var, 'RainSum', 'T', 'SM'],  # test of ZA
      'rs': [bio_var, bio_var + 'max'],
      'rs_reduced': [bio_var],
      'rs_sm_reduced': [bio_var, 'SM'],
      'met': [rad_var, 'RainSum', 'T', 'Tmin', 'Tmax'],
      'met_reduced': [rad_var, 'RainSum', 'T'],
      'met_sm_reduced': [rad_var, 'RainSum', 'T', 'SM']
    }

    # dictionary for group labels used in plots
    self.feature_groups2labels = {
      'rs_met': 'RS&Met',
      'rs_met_reduced': 'RS&Met-',
      'rs_met_sm_reduced': 'SM&RS&Met-',
      'rs': 'RS',
      'rs_reduced': 'RS-',
      'rs_sm_reduced': 'SM&RS-',
      'met': 'Met',
      'met_reduced': 'Met-',
      'met_sm_reduced': 'SM&Met-',
      'maxRS_met': 'maxRS&Met',
      'maxRS_met_reduced': 'maxRS&Met-',
      'maxRS_met_sm_reduced': 'SM&maxRS&Met-',
      'maxRS': 'maxRS',
      'maxRS_sm': 'SM&maxRS'
    }

    # model configuration settings to be tested
    self.time_samplings = ['M']  # ["M"]  # ['P', 'M']

    # # y variables to be predicted
    #self.yvars = ['Yield']  # ['Yield', 'Production']

    # Admin unit IDs OHE types to be tested
    self.doOHEs = ['none', 'AU_level']  # ['none', 'AU_level', 'Cluster_level']

    # trend
    self.addYieldTrend = [True, False]
    # number of years of y data before the first year with features AND number of years for trend computation
    self.ny_max_trend = 12

    # Add average y to feature set
    # An alternative way to pass the admin-level unobserved effect to the model. The y average by AU is used as an
    # additional feature. {old: In the outer CV leave one year out loop, it uses the mean of training y}
    # always scaled because features are scaled. It is alternative to DoOHEnc # (to be used only with DoOHEnc set to False).
    # Admitted values: False, True
    self.AddTargetMeanToFeature = False

    # Feature selection
    self.feature_selections = ['none', 'MRMR']
    # percentage of features to be selected (as grid to be tested)
    self.feature_prct_grid = [5, 25, 50, 75]

    # Data reduction with PCA
    self.dataReduction = ['none', 'PCA']
    self.PCAprctVar2keep = 90

    # Hyperparameters grid space
    self.hyperparopt = 'grid'
    if self.hyperparopt == 'grid':
      # Hidden layers treated as hyperparameters for NN MLP (1, 2 and 3 layers)
      # Set values that are exponents of two or values that can be divided by two
      hl2 = [(i, j) for i in [16, 32, 48, 64] for j in [16, 32, 48, 64]]
      hl3 = [[16, 32, 16], [16, 48, 16], [32, 48, 32], [32, 64, 32], [48, 64, 48], [32, 32, 32], [48, 48, 48],
             [64, 64, 64],
             [16, 16, 16]]
      hl = hl2 + hl3
      #self.hyperGrid = dict(Lasso={'alpha': np.logspace(-5, 0, 13, endpoint=True).tolist()},
      self.hyperGrid = dict(Lasso={'alpha': np.logspace(-3, 0.5, 15, endpoint=True).tolist()}, # change in 27/6/2024 to be > 0 and also >1 (up to 3)
                       RandomForest={'max_depth': [10, 15, 20, 25, 30, 35, 40],  # Maximum number of levels in tree
                                     #'max_features': ['auto', 'sqrt'],  # Number of features to consider at every split
                                     'max_features': ['1', 'sqrt'],  # Number of features to consider at every split # change in 27/6/2024
                                     #'n_estimators': [100, 250, 500],  # Number of trees in random forest
                                     'n_estimators': [50, 100, 250, 500],  # Number of trees in random forest # change in 27/6/2024
                                     #'min_samples_split': np.linspace(0.2, 0.8, 6, endpoint=True).tolist() # change in 27/6/2024 back to default
                                     },
                       #MLP={'alpha': np.logspace(-5, -1, 6, endpoint=True),
                        MLP={'alpha': np.logspace(-5, -2, 5, endpoint=True), # change in 27/6/2024
                            'hidden_layer_sizes': hl,
                            #'activation': ['relu', 'tanh']  hange in 27/6/2024 back to default

                            'learning_rate': ['constant', 'adaptive']},

                       # SVR_linear={'gamma': np.logspace(-2, 2, 2, endpoint=True).tolist(),
                       #                  # gamma defines how much influence a single training example has.
                       #                  # The larger gamma is, the closer other examples must be to be affected.
                       #                  'epsilon': np.logspace(-6, .5, 2, endpoint=True).tolist(),
                       #                  # 'epsilon': np.logspace(-6, .5, 7, endpoint=True).tolist(),
                       #                  'C': [1, 100]},
                       SVR_linear={#'gamma': np.logspace(-2, 2, 7, endpoint=True).tolist(), # change in 27/6/2024: there is no gamma in linear
                                   # gamma defines how much influence a single training example has.
                                   # The larger gamma is, the closer other examples must be to be affected.
                                   #'epsilon': np.logspace(-6, .5, 7, endpoint=True).tolist(),
                                   'epsilon': np.logspace(-4, .5, 7, endpoint=True).tolist(), # change in 27/6/2024
                                   #'C': [1e-5, 1e-4, 1e-3, 1e-2, 1, 10, 100]}, # change in 27/6/2024
                                   'C': [0.001, 0.01, 0.1, 1, 10, 100, 300]}, #thi sis log scale  plus 200
                       # SVR_rbf={'gamma': np.logspace(-2, 2, 2, endpoint=True).tolist(),
                       #          'epsilon': np.logspace(-6, .5, 2, endpoint=True).tolist(),
                       #          'C': [1e-5, 100]},
                       SVR_rbf={'gamma': np.logspace(-3, 1, 7, endpoint=True).tolist(),
                                     'epsilon': np.logspace(-4, .5, 7, endpoint=True).tolist(), # change in 27/6/2024
                                     #'C': [1e-5, 1e-4, 1e-3, 1e-2, 1, 10, 100]},
                                     'C':[0.001, 0.01, 0.1, 1, 10, 100, 300]}, # change in 27/6/2024
                            #GPR1={'alpha': [1e-10, 1e-5, 1e-1]},
                       #GPR2={'alpha': [1e-10, 1e-5, 1e-1, 0.05]})
                       GPR = {'alpha': [1e-10, 1e-7, 1e-4, 0.01, 0.1, 0.5]},
                       GBR={'learning_rate': [0.01, 0.05, 0.1],
                           # Empirical evidence suggests that small values of learning_rate favor better test error.
                            # [HTF] recommend to set the learning rate to a small constant (e.g. learning_rate <= 0.1)
                            # and choose n_estimators by early stopping.
                           'max_depth': [10, 20, 40],
                           'n_estimators': [100, 250, 500],
                           'min_samples_split': np.linspace(0.1, 0.8, 6, endpoint=True).tolist()},
                       # https://stackoverflow.com/questions/69786993/tuning-xgboost-hyperparameters-with-randomizedsearchcv
                       XGBoost = {'learning_rate': [0.05, 0.1, 0.2],
                                  'max_depth': [3, 6, 9],
                                  'min_child_weight': [1, 5, 10],
                                  # 'gamma': [0, 1, 2, 4],
                                  # 'lambda': [0.5, 1, 2, 4],
                                  # 'subsample': [0.25, 0.5, 0.75, 1.0],
                                  'n_estimators': [50, 100, 400, 800]} #new 2024
                                  #'gamma': [1, 2, 4, 6]}
                                 #'n_estimators': [50, 100, 250, 500],
                       )

def config_reducer(modelSettings, run_name):
    # MODIFY HERE TO DO LESS TESTING
    # 'feature_groups': {
    # 'rs_met': ['ND', 'NDmax', 'rad', 'RainSum', 'T', 'Tmin', 'Tmax'], 'rs_met_reduced': ['ND', 'RainSum', 'T'],
    # 'rs_met_sm_reduced': ['ND', 'RainSum', 'T', 'SM'], 'rs': ['ND', 'NDmax'], 'rs_reduced': ['ND'],
    # 'rs_sm_reduced': ['ND', 'SM'], 'met': ['rad', 'RainSum', 'T', 'Tmin', 'Tmax'],
    # 'met_reduced': ['rad', 'RainSum', 'T'], 'met_sm_reduced': ['rad', 'RainSum', 'T', 'SM']
    if run_name == 'test_change_names_changed':
        want_keys = ['rs_sm_reduced']
        modelSettings.feature_groups = dict(filter(lambda x: x[0] in want_keys, modelSettings.feature_groups.items()))
        modelSettings.doOHEs = ['AU_level']
        modelSettings.feature_prct_grid = [5, 25, 50, 100]
        want_keys = ['Lasso'] #used in run month5
        #want_keys = ['XGBoost'] # used in run month5 month5_onlyXGB
        modelSettings.hyperGrid = dict(filter(lambda x: x[0] in want_keys, modelSettings.hyperGrid.items()))
        modelSettings.feature_selections = ['none']
        modelSettings.addYieldTrend = [False]
        modelSettings.dataReduction = ['none']
    elif run_name == 'MA_20250512':
        want_keys = ['Lasso', 'XGBoost', 'SVR_linear', 'SVR_rbf']
        modelSettings.hyperGrid = dict(filter(lambda x: x[0] in want_keys, modelSettings.hyperGrid.items()))
    elif run_name == 'buttami':
        want_keys = ['Lasso']
        modelSettings.hyperGrid = dict(filter(lambda x: x[0] in want_keys, modelSettings.hyperGrid.items()))
        want_keys = ['rs_sm_reduced']
        modelSettings.feature_groups = dict(filter(lambda x: x[0] in want_keys, modelSettings.feature_groups.items()))
        modelSettings.doOHEs = ['AU_level']
        modelSettings.feature_selections = ['none']
        modelSettings.addYieldTrend = [False]
        modelSettings.dataReduction = ['none']
    else: #some default
        want_keys = ['Lasso', 'GPR', 'XGBoost', 'SVR_linear', 'SVR_rbf']
        modelSettings.hyperGrid = dict(filter(lambda x: x[0] in want_keys, modelSettings.hyperGrid.items()))
        # want_keys = ['rs_met_sm_reduced',  'rs',  'rs_reduced', 'rs_sm_reduced', 'met_sm_reduced']
    return modelSettings