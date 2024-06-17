import time
import os
import glob
from pathlib import Path
from A_config import a10_config
from C_model_setting import c100_save_model_specs
from B_preprocess import b100_load
from D_modelling import d090_model_wrapper
from F_post_processsing import F100_gather_hindcast_output

if __name__ == '__main__':
    '''
    This script shall be used to for:
    tuning models with double LOYO loop (test various configuration)
    
    PART A is run locally to generate data and spec files
    PART B can be run locally or on HT Condor
    '''
    # USER SETTINGS ###########################################################
    # Give a name to the run that will be used to make the output dir name
    run_name = 'test0' #
    # config file to be used
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZAsummer/ZAsummer_config.json'
    config_fn = r'V:\foodsec\Projects\SNYF\ZA_test_new_code\ZAsummer_config.json'
    # specify months on which to forecast
    forecastingMonths = [5]
    # Use condor or run locally
    tune_on_condor = False
    if tune_on_condor:
        dir_condor_submit = '/eos/jeodpp/data/projects/ML4CAST/'
    else:
        dir_condor_submit = r'V:\foodsec\Projects\SNYF\ZA_test_new_code'
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
    spec_files_list = glob.glob(os.path.join(config.models_spec_dir, '*.json'))
    if tune_on_condor == False:
        # get the produced spec file list
        for fn in spec_files_list:
            d090_model_wrapper.fit_and_validate_single_model(fn, config, runType)
        F100_gather_hindcast_output.gather_output(config.models_out_dir)
        print("--- %s seconds ---" % (time.time() - start_time))
        print('ended ZA_manager')
    else:
        # running with condor
        # make the task list (id, filename full path)
        condor_task_list_fn = os.path.join(config.root_dir, 'HT_condor_task_arguments.txt')
        if os.path.exists(condor_task_list_fn):
            os.remove(condor_task_list_fn)
        f_obj = open(condor_task_list_fn, 'a')
        for el in spec_files_list:
            id =config.AOI + '_' + os.path.splitext(os.path.basename(el))[0]
            f_obj.write(f'{id} {str(el)} {config_fn} {run_name}\n')
        f_obj.close()

        # launch the condor processing
        #Petar:
        # adjust the temmplate
        with open('G_HTCondor/condor.submit_template') as tmpl:
            content = tmpl.read()
            content = content.format(AOI=config.AOI, root_dir=config.root_dir)
        with open(os.path.join(dir_condor_submit, 'condor.submit'), 'w') as out:
            out.write(content)
        #debug
        # from G_HTCondor import condor_launcher
        # condor_launcher.launcher(r'V:\foodsec\Projects\SNYF\ZA_test_new_code\MLYF\RUN_test0_TUNING\Specs\000010_Maize_total_Lasso.json', config_fn, run_name)

        # sudo -u ml4castproc condor_submit condor.submit
        # place the one above somewhere USE FULL PATH for 'condor.submit'
        run_cmd = ['sudo', '-u', 'ml4castproc', 'condor_submit', 'condor.submit']

        # Check below with Petar
        # p = subprocess.run(run_cmd, shell=False, input='\n', capture_output=True, text=True)
        # if p.returncode != 0:
        #     self.log('ERR', p.stderr)
        #     raise Exception('Step subprocess error')


