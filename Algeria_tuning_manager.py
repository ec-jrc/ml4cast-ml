import time
import os
import glob
import json

from A_config import a10_config
from C_model_setting import c100_save_model_specs
from B_preprocess import b100_load
from D_modelling import d100_modeller
import time

if __name__ == '__main__':
    '''
    This script shall be used to run these different runTypes: 
    1) [tuning] tunes models with double LOYO loop (test various configuration)
    2) [opeTune] tunes the best model on all available years
    3) [opeForecast] run the operational yield of 2) using predictors only. Predictors are save in a
       different dir that can be updated and to avoid overwrite of features used for training
    '''

    start_time = time.time()
    # load region specific data info
    config = a10_config.read(r'V:\foodsec\Projects\SNYF\Algeria\MLYF\Algeria_config.json')
    # load model configurations to be tested
    modelSettings = a10_config.mlSettings(forecastingMonths = [7])
    runType = 'tuning'  # ['tuning', 'opeForecast']

    if runType == 'tuning':
        ###################################################################################################################
        # MODIFY HERE TO DO LESS TESTING
        want_keys = ['rs_met_reduced']
        modelSettings.feature_groups = dict(filter(lambda x: x[0] in want_keys, modelSettings.feature_groups.items()))
        modelSettings.feature_prct_grid = [50, 75, 100]
        want_keys = ['Lasso', 'SVR_linear']
        modelSettings.hyperGrid = dict(filter(lambda x: x[0] in want_keys, modelSettings.hyperGrid.items()))
        modelSettings.addYieldTrend = [False]
        modelSettings.dataReduction = ['none']
        ###################################################################################################################

    if False : #already tested
        b100_load.LoadPredictors_Save_Csv(config, runType)
        b100_load.build_features(config, runType)
        # remove admin units with missing data in yield !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        b100_load.LoadLabel_Exclude_Missing(config)
        # prepare json files specifying the details of each run to be tested
        c100_save_model_specs.save_model_specs(config, modelSettings)

    # print(config.__dict__)
    # print(config.sos)

    # get the produced spec file list
    spec_files_list = glob.glob(os.path.join(config.models_spec_dir, '*.json'))
    for fn in spec_files_list:
        with open(fn, 'r') as fp:
            uset = json.load(fp)
        #print(uset)
        hindcaster = d100_modeller.YieldModeller(uset)
        # preprocess
        X, y, groups, feature_names, AU_codes = hindcaster.preprocess(config, runType)
        tic = time.time()
        # fit and put results in a dict
        hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, prctPegged, \
            selected_features_names, prct_selected, n_selected, \
            avg_scoring_metric_on_val = hindcaster.fit(X, y, groups, feature_names, AU_codes)
        runTimeH = (time.time() - tic) / (60 * 60)
        print(f'Model fitted in {runTimeH} hours')
        hindcaster.validate(hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, prctPegged, runTimeH, feature_names, selected_features_names,
                        prct_selected, n_selected, avg_scoring_metric_on_val, config, save_file=True, save_figs=True)
    print('ended Algeria_manager')




    print("--- %s seconds ---" % (time.time() - start_time))


