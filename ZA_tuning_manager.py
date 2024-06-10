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
    This script shall be used to for:
    tuning models with double LOYO loop (test various configuration)
    
    PART A is run locally to generate data and spec files
    PART B can be run locally or on HT Condor
    '''
    # USER SETTINGS ###########################################################
    # Give a name to the run that will be used to make the output dir name
    run_name = 'testMic_with_missing' #
    # config file to be used
    config_fn = r'V:\foodsec\Projects\SNYF\ZA_test_new_code\ZAsummer_config.json'
    # specify months on which to forecast
    forecastingMonths = [5]
    # Use condor or run locally
    tune_on_condor = False
    # END OF USER SETTINGS ###########################################################

    # ----------------------------------------------------------------------------------------------------------
    # PART A
    runType = 'tuning'  # ['tuning', 'opeForecast']
    start_time = time.time()
    # load region specific data info
    config = a10_config.read(config_fn, run_name)
    # make necessary directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.models_dir).mkdir(parents=True, exist_ok=True)
    Path(config.models_spec_dir).mkdir(parents=True, exist_ok=True)
    Path(config.models_out_dir).mkdir(parents=True, exist_ok=True)
    # load model configurations to be tested
    modelSettings = a10_config.mlSettings(forecastingMonths=forecastingMonths) #[3,6]

    ##################################################################################################################
    # MODIFY HERE TO DO LESS TESTING
    #config.crops = ['Maize_total']
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
    if True:
        b100_load.LoadPredictors_Save_Csv(config, runType)
        b100_load.build_features(config, runType)
        # remove admin units with missing data in yield !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # b100_load.LoadLabel_Exclude_Missing(config)
        b100_load.LoadLabel(config)
        # prepare json files specifying the details of each run to be tested
        c100_save_model_specs.save_model_specs(config, modelSettings)
    # print(config.__dict__)

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


