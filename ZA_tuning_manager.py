import time
import os
import glob
import json
from pathlib import Path

from A_config import a10_config
from C_model_setting import c100_save_model_specs
from B_preprocess import b100_load
from D_modelling import d100_modeller
from F_post_processsing import F100_gather_hindcast_output
from G_HTCondor import g100_HTCondor

if __name__ == '__main__':
    '''
    This script shall be used to run these different runTypes: 
    1) [tuning] tunes models with double LOYO loop (test various configuration)
    2) [opeForecast] run the operational yield of 2) using predictors only. Predictors are saved in a
       different dir that can be updated and to avoid overwrite of features used for training
    PART A is run locally to generate data and spec files
    PART B can be run locally or on HT Condor
    '''


    # ----------------------------------------------------------------------------------------------------------
    # PART A
    # Give a name to the run that will be used to make the output dir name
    run_name = 'buttami'
    runType = 'tuning'  # ['tuning', 'opeForecast']
    tune_on_condor = False

    start_time = time.time()

    # load region specific data info
    config = a10_config.read(r'V:\foodsec\Projects\SNYF\ZA_test_new_code\ZAsummer_config.json', run_name)
    # make necessary directories
    if runType == 'tuning':
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.models_dir).mkdir(parents=True, exist_ok=True)
        Path(config.models_spec_dir).mkdir(parents=True, exist_ok=True)
        Path(config.models_out_dir).mkdir(parents=True, exist_ok=True)
    elif runType in ['opeTune', 'opeForecast']:
        Path(config.ope_run_dir).mkdir(parents=True, exist_ok=True)
    # load model configurations to be tested
    modelSettings = a10_config.mlSettings(forecastingMonths = [5]) #[3,6]


    if runType == 'tuning':
        ##################################################################################################################
        # MODIFY HERE TO DO LESS TESTING
        config.crops = ['Maize_total']
        want_keys = ['rs_met_reduced']
        modelSettings.feature_groups = dict(filter(lambda x: x[0] in want_keys, modelSettings.feature_groups.items()))
        modelSettings.doOHEs = ['AU_level']
        modelSettings.feature_selections = ['none']
        modelSettings.feature_prct_grid = [50]
        want_keys = ['Lasso'] #['GPR'] #['XGBoost']
        modelSettings.hyperGrid = dict(filter(lambda x: x[0] in want_keys, modelSettings.hyperGrid.items()))
        modelSettings.addYieldTrend = [False]
        modelSettings.dataReduction = ['none']
        ###################################################################################################################
    print(modelSettings.__dict__)

    if True: #already tested
        b100_load.LoadPredictors_Save_Csv(config, runType)
        b100_load.build_features(config, runType)
        # remove admin units with missing data in yield !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        b100_load.LoadLabel_Exclude_Missing(config)
        # prepare json files specifying the details of each run to be tested
        c100_save_model_specs.save_model_specs(config, modelSettings)

    # print(config.__dict__)
    # print(config.sos)

    # ----------------------------------------------------------------------------------------------------------
    # PART B
    if tune_on_condor == False:
        # get the produced spec file list
        spec_files_list = glob.glob(os.path.join(config.models_spec_dir, '*.json'))
        for fn in spec_files_list:
            g100_HTCondor.fit_and_validate_single_model(fn, config, runType)

        print('ended ZA_manager')

        F100_gather_hindcast_output.gather_output(config.models_out_dir)
        print("--- %s seconds ---" % (time.time() - start_time))
    else:
        # running with condor
        # make the list
        # launch the condor processing
        print('to be implemented')


# tic = time.time()
# with open(fn, 'r') as fp:
#     uset = json.load(fp)
# print(uset)
# hindcaster = d100_modeller.YieldModeller(uset)
# # preprocess
# X, y, groups, feature_names, AU_codes = hindcaster.preprocess(config, runType)
# # fit and put results in a dict
# hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, prctPegged, \
#     selected_features_names, prct_selected, n_selected, \
#     avg_scoring_metric_on_val, fitted_model = hindcaster.fit(X, y, groups, feature_names, AU_codes, runType)
# runTimeH = (time.time() - tic) / (60 * 60)
# print(f'Model fitted in {runTimeH} hours')
# # error stats
# hindcaster.validate(hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, prctPegged, runTimeH, feature_names, selected_features_names,
#                 prct_selected, n_selected, avg_scoring_metric_on_val, config, save_file=True, save_figs=False)
