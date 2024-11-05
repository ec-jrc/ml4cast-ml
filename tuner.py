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
from F_post_processsing import F100_analyze_hindcast_output

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

def write_time_info(config, run_name):
    fn_output = os.path.join(config.models_dir, 'time_info_readme_' + run_name + '.txt')
    with open(fn_output, 'a') as f:
        f.write('Year start: ' + str(config.year_start) + '\n')
        f.write('Year end: ' + str(config.year_end) + '\n')
        f.write('Forecasting months: ' + str(config.forecastingMonths) + '\n')
        f.write('SOS: ' + str(config.sos) + '\n')
        f.write('SOS month (month 1): ' + str(config.sosMonth) + '\n')
        f.write('EOS: ' + str(config.eos) + '\n')
        f.write('EOS month (last 1): ' + str(config.eosMonth) + '\n')
        f.write('Calendar_Month, Forecasting_Month, Progress_from_months' + '\n')
        if config.sosMonth < config.eosMonth:
            cal_months = list(range(int(config.sosMonth), int(config.eosMonth+1)))
        else:
            cal_months = list(range(int(config.sosMonth), 12 + 1)) + list(range(1, int(config.eosMonth) + 1))
        fm = 1
        for cm in cal_months:
            f.write(str(cm) + ', ' + str(fm) + ', ' + str(fm/len(cal_months)) + '\n')
            fm = fm + 1

def tuneA(run_name, config_fn, tune_on_condor, runType):
    """
    PART A is run locally to generate data and spec files
    PART B tune each of the spec file and produce the output. it can be run locally or on HT Condor depending on tune_on_condor
    """
    # ----------------------------------------------------------------------------------------------------------
    # PART A
    # load region specific data info
    config = a10_config.read(config_fn, run_name, run_type=runType)
    forecastingMonths = config.forecastingMonths
    # make necessary directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.models_dir).mkdir(parents=True, exist_ok=True)
    Path(config.models_spec_dir).mkdir(parents=True, exist_ok=True)
    Path(config.models_out_dir).mkdir(parents=True, exist_ok=True)

    # load model configurations to be tested
    modelSettings = a10_config.mlSettings(forecastingMonths=forecastingMonths) #[3,6]


    ##################################################################################################################
    # MODIFY in this function TO DO LESS TESTING
    modelSettings = a10_config.config_reducer(modelSettings, run_name)
    # save model setting and config in the run dir (config.output_dir)
    with open(os.path.join(config.models_dir, run_name + '_model_settings.json'), 'w') as fp:
        json.dump(modelSettings.__dict__, fp, indent=4)
    with open(os.path.join(config.models_dir, run_name + '_config.json'), 'w') as fp:
        json.dump(config.__dict__, fp, indent=4)
    ###################################################################################################################
    print(modelSettings.__dict__)

    # write time information (sos, eos, months of tuning) in a time readme file in the tune directory
    write_time_info(config, run_name)

    # Prepare input files
    b100_load.LoadPredictors_Save_Csv(config, runType)
    b100_load.build_features(config, runType)
    b100_load.LoadLabel(config)
    # prepare json files specifying the details of each run to be tested
    c100_save_model_specs.save_model_specs(config, modelSettings)
    # print(config.__dict__)
    spec_files_list = glob.glob(os.path.join(config.models_spec_dir, '*.json'))
    return spec_files_list

def checkExistingSubmit(condor_task_list_base_name, condor_task_list_fn, config, spec_files_list):
    # If the file already exists it means that we are re-running Condor for some reasons (get stopped somehow)
    # In this case copy the HT_condor_task_arguments to HT_condor_task_arguments_all and keep
    # only the entries that were not successful (sya that logs will be overwritten and must be moved a folder,
    # wait for Y from keyboard)

    print(condor_task_list_fn)
    print('HT_condor_task_arguments.txt already exists')
    print('>Jobs that produced already outputs will be skipped')
    print('Log files will be deleted, move to a named dir if you want to keep them')
    pro = input('Type Y to proceed\n')
    if pro == 'Y':
        fn_rename = condor_task_list_base_name.split('.')[0] + '_all.txt'
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
        scount = 0
        for s in spec_files_list:
            print(str(scount), s)
            scount = scount + 1
        return spec_files_list
    else:
        sys.exit('Tuner terminated by user')
def tuneB(run_name, config_fn, tune_on_condor, runType, spec_files_list):
    # ----------------------------------------------------------------------------------------------------------
    # new version that splits big lists of task > 7500 in two blocks, 5000 and the rest
    # PART B
    nMaxTask = 7500
    start_time = time.time()
    config = a10_config.read(config_fn, run_name, run_type=runType)

    if tune_on_condor == False:
        # get the produced spec file list
        for fn in spec_files_list:
            print(fn)
            d090_model_wrapper.fit_and_validate_single_model(fn, config, runType)
        #F100_analyze_hindcast_output.gather_output(config.models_out_dir)
        print("--- %s seconds ---" % (time.time() - start_time))
    else:
        # running with condor
        dir_condor_submit = config.models_dir
        # make the task list (id, filename full path)
        condor_task_list_base_name = 'HT_condor_task_arguments.txt'
        condor_task_list_fn = os.path.join(config.models_dir, condor_task_list_base_name)
        spec_files_list1 = spec_files_list
        if len(spec_files_list) > nMaxTask:
            spec_files_list1 = spec_files_list[0:7500]
            spec_files_list2 = spec_files_list[7500:-1]
            condor_task_list_base_name2 = 'HT_condor_task_arguments2.txt'
            condor_task_list_fn2 = os.path.join(config.models_dir, condor_task_list_base_name2)
        # If the file already exists it means that we are re-running Condor for some reasons (get stopped somehow)
        # In this case copy the HT_condor_task_arguments to HT_condor_task_arguments_all and keep
        # only the entries that were not successful (sya that logs will be overwritten and must be moved a folder,
        # wait for Y from keyboard)
        if os.path.isfile(condor_task_list_fn):
            spec_files_list2 = checkExistingSubmit(condor_task_list_base_name, condor_task_list_fn, config, spec_files_list2)
        if len(spec_files_list) > nMaxTask:
            # there was a second ht condor
            if os.path.isfile(condor_task_list_fn2):
                spec_files_list1 = checkExistingSubmit(condor_task_list_base_name2, condor_task_list_fn2, config, spec_files_list)


        if len(spec_files_list1) == 0 and len(spec_files_list2) == 0:
            print('No files to re-run, execution will stop')
            sys.exit()
        # spec_files_list1 exists if I am here
        if len(spec_files_list1) > 0:
            if os.path.exists(condor_task_list_fn):
                os.remove(condor_task_list_fn)
            f_obj = open(condor_task_list_fn, 'a')
            for el in spec_files_list1:
                f_obj.write(f'{str(el)} {config_fn} {run_name} {runType}\n')
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

            if len(spec_files_list) > nMaxTask:
                # the task was big there is another task to run, but we need to wait an hour
                time.sleep(60 * 60)
                if len(spec_files_list2) > 0:
                    if os.path.exists(condor_task_list_fn2):
                        os.remove(condor_task_list_fn2)
                    f_obj = open(condor_task_list_fn2, 'a')
                    for el in spec_files_list2:
                        f_obj.write(f'{str(el)} {config_fn} {run_name} {runType}\n')
                    f_obj.close()
                    # Make sure that the run.sh in this project is executable (# chmod 755 run.sh)
                    # adjust the condor.submit template
                    condSubPath = os.path.join(dir_condor_submit, 'condor.submit2')
                    with open('G_HTCondor/condor.submit_template') as tmpl:
                        content = tmpl.read()
                        content = content.format(AOI=config.AOI,
                                                 root_dir=config.models_dir)  # , shDestination=shDestination)
                    with open(condSubPath, 'w') as out:
                        out.write(content)
                    # Make the dirs for condor output on /mnt/jeoproc/log/ml4castproc/, clean content
                    Path(os.path.join('/mnt/jeoproc/log/ml4castproc', config.AOI, 'out')).mkdir(parents=True,
                                                                                                exist_ok=True)
                    Path(os.path.join('/mnt/jeoproc/log/ml4castproc', config.AOI, 'err')).mkdir(parents=True,
                                                                                                exist_ok=True)
                    Path(os.path.join('/mnt/jeoproc/log/ml4castproc', config.AOI, 'log')).mkdir(parents=True,exist_ok=True)
                    if len(spec_files_list1) == 0:
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


def tuneBOLD(run_name, config_fn, tune_on_condor, runType, spec_files_list):
    # ----------------------------------------------------------------------------------------------------------
    # PART B
    start_time = time.time()
    config = a10_config.read(config_fn, run_name, run_type=runType)

    if tune_on_condor == False:
        # get the produced spec file list
        for fn in spec_files_list:
            print(fn)
            d090_model_wrapper.fit_and_validate_single_model(fn, config, runType)
        #F100_analyze_hindcast_output.gather_output(config.models_out_dir)
        print("--- %s seconds ---" % (time.time() - start_time))
    else:
        # running with condor
        dir_condor_submit = config.models_dir
        # make the task list (id, filename full path)
        condor_task_list_base_name = 'HT_condor_task_arguments.txt'
        condor_task_list_fn = os.path.join(config.models_dir, condor_task_list_base_name)
        # If the file already exists it means that we are re-running Condor for some reasons (get stopped somehow)
        # In this case copy the HT_condor_task_arguments to HT_condor_task_arguments_all and keep
        # only the entries that were not successful (sya that logs will be overwritten and must be moved a folder,
        # wait for Y from keyboard)
        if os.path.isfile(condor_task_list_fn):
            print('HT_condor_task_arguments.txt already exists')
            print('>Jobs that produced already outputs will be skipped')
            print('Log files will be deleted, move to a named dir if you want to keep them')
            pro = input('Type Y to proceed\n')
            if pro == 'Y':
                fn_rename = condor_task_list_base_name.split('.')[0] + '_all.txt'
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
                scount = 0
                for s in spec_files_list:
                    print(str(scount), s)
                    scount = scount + 1
            else:
                sys.exit('Tuner terminated by user')
            if len(new_file_list) == 0:
                print('No files to re-run, execution will stop')
                sys.exit()
        if os.path.exists(condor_task_list_fn):
            os.remove(condor_task_list_fn)
        f_obj = open(condor_task_list_fn, 'a')
        for el in spec_files_list:
            f_obj.write(f'{str(el)} {config_fn} {run_name} {runType}\n')
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





