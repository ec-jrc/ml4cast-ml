import time
import os
import sys
import glob
import json
from pathlib import Path
import subprocess
from A_config import a10_config
from C_model_setting import c100_save_model_specs
from B_preprocess import b100_load
from D_modelling import d090_model_wrapper
from F_post_processsing import F100_gather_hindcast_output
def remove_files(path):
  """
  Removes all files in the given directory path, but leaves subdirectories and their contents untouched.

  Args:
      path: The directory path from which to remove files.
  """
  for filename in os.listdir(path):
    filepath = os.path.join(path, filename)
    if os.path.isfile(filepath):
      os.remove(filepath)
    else:
      # Skip directories
      pass



def tune(run_name, config_fn, forecastingMonths, tune_on_condor):
    '''
    This script shall be used to for:
    tuning models with double LOYO loop (test various configuration)
    run_name: give a name to the run that will be used to make the output dir name
    # config file to be used
    config_fn: config file to be used
    forecastingMonths: specify months on which to forecast [5]
    tune_on_condor = Use condor or run locally, True or False
    
    PART A is run locally to generate data and spec files
    PART B can be run locally or on HT Condor
    '''
    # USER SETTINGS ###########################################################

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
    if tune_on_condor:
        dir_condor_submit = config.models_dir
    # load model configurations to be tested
    modelSettings = a10_config.mlSettings(forecastingMonths=forecastingMonths) #[3,6]
    ##################################################################################################################
    # MODIFY in this function TO DO LESS TESTING
    a10_config.config_reducer(modelSettings, run_name)
    ###################################################################################################################
    print(modelSettings.__dict__)
    # Prepare input files
    b100_load.LoadPredictors_Save_Csv(config, runType)
    b100_load.build_features(config, runType)
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
    else:
        # running with condor
        # make the task list (id, filename full path)
        fn_condor_task_list = 'HT_condor_task_arguments.txt'
        condor_task_list_fn = os.path.join(config.models_dir, fn_condor_task_list)
        # If the file already exists it means that we are re-running Condor for some reasons (get stopped somehow)
        # In this case copy the HT_condor_task_arguments to HT_condor_task_arguments_all and keep
        # only the entries that were not successful (sya that logs will be overwritten and must be moved a folder,
        # wait for Y from keyboard)
        if os.path.isfile(condor_task_list_fn):
            print('HT_condor_task_arguments.txt already exists')
            print('>Jobs that produced already outputs will be skipped')
            print('Log files will be deleted, move to a named dir if you want to keep them')
            pro = input('Type Y to proceed')
            if pro == 'Y':
                fn_rename = fn_condor_task_list.split('.')[0] + '_all.txt'
                os.rename(condor_task_list_fn, os.path.join(config.models_dir, fn_rename))
                # update spec_files_list
                new_file_list = []
                for el in spec_files_list:
                    # make the expected output name
                    with open(el, 'r') as fp:
                        uset = json.load(fp)
                    myID = uset['runID']
                    myID = f'{myID:06d}'
                    fn_to_check = os.path.join(config.models_out_dir, 'ID_' + str(myID) +
                                          '_crop_' + uset['crop'] + '_Yield_' + uset['algorithm'] + '_output.csv')
                    if not os.path.isfile(fn_to_check):
                        new_file_list.append(el)
                spec_files_list = new_file_list
                print('List of files with no output:')
                print(*spec_files_list, sep='\n')
            else:
                sys.exit('Tuner terminated by user')

        if os.path.exists(condor_task_list_fn):
            os.remove(condor_task_list_fn)
        f_obj = open(condor_task_list_fn, 'a')
        for el in spec_files_list:
            f_obj.write(f'{str(el)} {config_fn} {run_name}\n')
        f_obj.close()
        # Make sure that the run.sh in this project is executable (# chmod 755 run.sh)

        # adjust the condor.submit template
        condSubPath = os.path.join(dir_condor_submit, 'condor.submit')
        with open('G_HTCondor/condor.submit_template') as tmpl:
            content = tmpl.read()
            content = content.format(AOI=config.AOI, root_dir=config.models_dir) #, shDestination=shDestination)
        with open(condSubPath, 'w') as out:
            out.write(content)
        # Make the dirs for condor output on /mnt/jeoproc/log/ml4castproc/, clean content
        Path(os.path.join('/mnt/jeoproc/log/ml4castproc', config.AOI, 'out')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join('/mnt/jeoproc/log/ml4castproc', config.AOI, 'err')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join('/mnt/jeoproc/log/ml4castproc', config.AOI, 'log')).mkdir(parents=True, exist_ok=True)
        remove_files(os.path.join('/mnt/jeoproc/log/ml4castproc', config.AOI, 'out'))
        remove_files(os.path.join('/mnt/jeoproc/log/ml4castproc', config.AOI, 'err'))
        remove_files(os.path.join('/mnt/jeoproc/log/ml4castproc', config.AOI, 'log'))
        # Launch condor (sudo -u ml4castproc condor_submit condor.submit)
        run_cmd = ['sudo', '-u', 'ml4castproc', 'condor_submit', condSubPath]
        p = subprocess.run(run_cmd, shell=False, input='\n', capture_output=True, text=True)
        if p.returncode != 0:
            print('ERR', p.stderr)
            raise Exception('Step subprocess error')
        print('Batch submitted by tuner: ' + 'ml4cast_' + config.AOI)
        print(p.stdout)





