#import pandas as pd
import numpy as np
import os
import json


class read:
  def __init__(self, full_path_config, run_name):

    with open(full_path_config, 'r') as fp:
        jdict = json.load(fp)
    self.AOI = jdict['AOI']
    self.year_start = int(jdict['year_start'])
    self.year_end = int(jdict['year_end'])
    self.crops = jdict['crops']
    self.root_dir = jdict['root_dir']
    self.data_dir = os.path.join(self.root_dir, jdict['data_dir'])
    self.ope_data_dir = os.path.join(self.root_dir, jdict['ope_data_dir'])
    self.output_dir = os.path.join(self.root_dir, jdict['output_dir'])

    # run_stamp = datetime.datetime.today().strftime('%Y%m%d')
    self.ope_run_dir = os.path.join(self.output_dir, 'RUN_'+run_name +'_OPE')
    self.ope_run_out_dir = os.path.join(self.ope_run_dir, 'output')
    self.models_dir = os.path.join(self.output_dir, 'RUN_'+ run_name + '_TUNING')
    self.models_spec_dir = os.path.join(self.models_dir, 'Specs')
    self.models_out_dir = os.path.join(self.models_dir, 'Output')

    self.ivars = jdict['ivars']
    self.ivars_short = jdict['ivars_short']
    self.ivars_units = jdict['ivars_units']
    self.sos = int(jdict['sos'])
    self.eos = int(jdict['eos'])
    self.yield_units = jdict['yield_units']
    self.area_unit =jdict['area_unit']
    # factor that divide production values to get production in desired units
    self.production_scaler =jdict['production_scaler']



class mlSettings:
  def __init__(self, forecastingMonths):
    # Define settings used in the ML workflow

    #set forcasting month (1 is the first)
    self.forecastingMonths = forecastingMonths

    # scikit, numbers of cores to be used when multi-thread is possible, at least 4
    self.nJobsForGridSearchCv = 4

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

    # feature groups to be considered
    rad_var = 'rad' #sometimes is 'Rad'
    self.feature_groups = {
      'rs_met': ['ND', 'NDmax', rad_var, 'RainSum', 'T', 'Tmin', 'Tmax'],
      'rs_met_reduced': ['ND', 'RainSum', 'T'],
      'rs_met_sm_reduced': ['ND', 'RainSum', 'T', 'SM'],  # test of ZA
      'rs': ['ND', 'NDmax'],
      'rs_reduced': ['ND'],
      'rs_sm_reduced': ['ND', 'SM'],
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
      'met_sm_reduced': 'SM&Met-'
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
    self.feature_prct_grid = [5, 25, 50, 75, 100]

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
      self.hyperGrid = dict(Lasso={'alpha': np.logspace(-5, 0, 13, endpoint=True).tolist()},
                       RandomForest={'max_depth': [10, 15, 20, 25, 30, 35, 40],  # Maximum number of levels in tree
                                     'max_features': ['auto', 'sqrt'],  # Number of features to consider at every split
                                     'n_estimators': [100, 250, 500],  # Number of trees in random forest
                                     'min_samples_split': np.linspace(0.2, 0.8, 6, endpoint=True).tolist()},
                       MLP={'alpha': np.logspace(-5, -1, 6, endpoint=True),
                            'hidden_layer_sizes': hl,
                            'activation': ['relu', 'tanh'],
                            'learning_rate': ['constant', 'adaptive']},

                       # SVR_linear={'gamma': np.logspace(-2, 2, 2, endpoint=True).tolist(),
                       #                  # gamma defines how much influence a single training example has.
                       #                  # The larger gamma is, the closer other examples must be to be affected.
                       #                  'epsilon': np.logspace(-6, .5, 2, endpoint=True).tolist(),
                       #                  # 'epsilon': np.logspace(-6, .5, 7, endpoint=True).tolist(),
                       #                  'C': [1, 100]},
                       SVR_linear={'gamma': np.logspace(-2, 2, 7, endpoint=True).tolist(),
                                   # gamma defines how much influence a single training example has.
                                   # The larger gamma is, the closer other examples must be to be affected.
                                   'epsilon': np.logspace(-6, .5, 7, endpoint=True).tolist(),
                                   'C': [1e-5, 1e-4, 1e-3, 1e-2, 1, 10, 100]},
                       # SVR_rbf={'gamma': np.logspace(-2, 2, 2, endpoint=True).tolist(),
                       #          'epsilon': np.logspace(-6, .5, 2, endpoint=True).tolist(),
                       #          'C': [1e-5, 100]},
                       SVR_rbf={'gamma': np.logspace(-2, 2, 7, endpoint=True).tolist(),
                                     'epsilon': np.logspace(-6, .5, 7, endpoint=True).tolist(),
                                     'C': [1e-5, 1e-4, 1e-3, 1e-2, 1, 10, 100]},
                            #GPR1={'alpha': [1e-10, 1e-5, 1e-1]},
                       #GPR2={'alpha': [1e-10, 1e-5, 1e-1, 0.05]})
                       GPR = {'alpha': [1e-10, 1e-5, 1e-1, 0.05]},
                       GBR={'learning_rate': [0.01, 0.05, 0.1],
                           # Empirical evidence suggests that small values of learning_rate favor better test error.
                            # [HTF] recommend to set the learning rate to a small constant (e.g. learning_rate <= 0.1)
                            # and choose n_estimators by early stopping.
                           'max_depth': [10, 20, 40],
                           'n_estimators': [100, 250, 500],
                           'min_samples_split': np.linspace(0.1, 0.8, 6, endpoint=True).tolist()},
                       XGBoost = {'learning_rate': [0.05, 0.1, 0.3, 0.5],
                                  'max_depth': [2, 4, 6, 8],
                                  'min_child_weight': [1, 3, 5]}
                                  #'gamma': [0, 2, 5]}
                       )
