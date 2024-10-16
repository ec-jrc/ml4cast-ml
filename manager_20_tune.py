import tuner
import datetime
import time
import glob
import os
import sys
import json
import threading
import subprocess
import pandas as pd
import manager_0_user_params as upar
from A_config import a10_config



def monitor_condor_q(time_step_minutes, submitter, config, run_name):
  start_time = time.time()
  first_check = True
  fn_output = os.path.join(config.models_dir, 'monitor_condor_q_RUN_' + run_name + '.txt')
  with open(fn_output, 'w') as f:
    f.write('Condor monitoring for run ' + run_name + ' ' + str(datetime.datetime.now()) + '\n')
  while True:
    # JobStatus is an integer;
    # states: - 1: Idle(I) - 2: Running(R) - 3: Removed(X) - 4: Completed(C) - 5: Held(H) - 6: Transferring
    # Output - 7: Suspended
    # sudo -u ml4castproc condor_q submitter ml4castproc -format '%s ' JobBatchName -format '%-3d ' ProcId -format '%-3d\n' JobStatus
    # check_cmd = ['sudo', '-u', submitter, 'condor_q', 'submitter', submitter, '-format', "'%s '", 'JobBatchName', '-format', "'%-3d '", 'ProcId', '-format', r"'%-3d\n'", 'JobStatus']
    check_cmd = ['sudo', '-u', submitter, 'condor_q', 'submitter', submitter, '-format', "%s ", 'JobBatchName',
                 '-format', "%-3d ", 'ProcId', '-format', "%s ", 'GlobalJobId','-format', r"%-3d\n", 'JobStatus']
    # Write output to a file
    with open(fn_output, 'a') as f:
        f.write(" ".join(check_cmd) + '\n')
        f.write('\n')
    p = subprocess.run(check_cmd, shell=False, input='\n', capture_output=True, text=True)
    if p.returncode != 0:
        with open(fn_output, 'a') as f:
            f.write('ERR ' + p.stderr + '\n')
            f.write('\n')
            print('ERR', p.stderr)
        #raise Exception('Step subprocess error')
    else:
        # make it a df
        # Split the string into lines
        lines = p.stdout.splitlines()
        # Define a list to store data for the DataFrame
        data_list = []
        # Loop through each line and extract relevant information
        for line in lines:
            # Split the line by whitespace (considering multiple spaces with `\s+`)
            parts = line.split()
            data_list.append([parts[0], int(parts[1]), parts[2], int(parts[3])])
        # Create the DataFrame with column names
        df = pd.DataFrame(data_list, columns=["BatchName", "ProcID", "GlobalJobId", "Status"])
        # map meaning of state
        statesDict = {1: 'Idle', 2: 'Running', 3: 'Removed', 4: 'Completed', 5: 'Held', 6: 'Transferring'}
        df['StatusString'] = df['Status'].map(statesDict)
        if len(df) == 0:
            with open(fn_output, 'a') as f:
                f.write("###################################" + '\n')
                f.write("nothing on Condor anymore, the monitoring will stop" + '\n')
                f.write("--- %s Hours ---" % str((time.time() - start_time)/(60*60)))
                f.write('\n')
            print('nothing on Condor anymore, the monitoring will stop')
            print("--- %s Hours ---" % str((time.time() - start_time)/(60*60)))
            # here add a check that all specs have a corresponding output
            spec_files_list = glob.glob(os.path.join(config.models_spec_dir, '*.json'))
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
            if len(new_file_list) > 0:
                with open(fn_output, 'a') as f:
                    f.write(str(len(new_file_list)) + ' files with no output:' + '\n')
                    f.write("List of files with no output (or to be rerun in case of previous fast tuning):" + '\n')
                    for i in new_file_list:
                        f.write(i + '\n')
                print(str(len(new_file_list)) + ' files with no output:')
                print('List of files with no output (or to be rerun in case of previous fast tuning):')
                print(*new_file_list, sep='\n')
            break
        with open(fn_output, 'a') as f:
            f.write('Condor stats on ' + str(datetime.datetime.now()) + '\n')
        # print('Condor stats on ' + str(datetime.datetime.now()))
        if first_check:
            first_check = False
            jobRequested = len(df)
        else:
            with open(fn_output, 'a') as f:
                f.write('Check at: ' + str(datetime.datetime.now()) + '\n')
                f.write('Jobs submitted: ' + str(jobRequested) + '\n')
                print('Check at: ' + str(datetime.datetime.now()))
                print('Jobs submitted: ' + str(jobRequested))
        with open(fn_output, 'a') as f:
            f.write('Jobs in que: ' + str(len(df)) + '\n')
            f.write('Jobs running: ' + str(len(df[df['StatusString']=='Running']))+ '\n')
            f.write('Jobs idle: ' + str(len(df[df['StatusString'] == 'Idle']))+ '\n')
            f.write('Jobs held: ' + str(len(df[df['StatusString'] == 'Held']))+ '\n')
        print('Jobs in que: ' + str(len(df)))
        print('Jobs running: ' + str(len(df[df['StatusString']=='Running'])))
        print('Jobs idle: ' + str(len(df[df['StatusString'] == 'Idle'])))
        print('Jobs held: ' + str(len(df[df['StatusString'] == 'Held'])))
        if len(df[df['StatusString'] == 'Held']) > 0:
            with open(fn_output, 'a') as f:
                f.write('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' + '\n')
                f.write(print('JOBS HELD') + '\n')
                f.write(f[df['StatusString'] == 'Held'].to_string() + '\n')
                f.write('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'+ '\n')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('JOBS HELD')
            print(df[df['StatusString'] == 'Held'])
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # print(df)
        # print('here')
        # Sleep for n minutes
    time.sleep(time_step_minutes*60)


if __name__ == '__main__':
    """
    Main function for tuning the models on a single pc or condor
    The user must define:
    run_name: a name that will be used to create output directory
    config_fn: the json file with essential info of the data (variables, phenology, crops to be included, etc)
    forecastingMonths: a list of forecasting times (remember that month 1 is the month of SOS)
    runType: can be "tuning" (double loop) or "fast_tuning" (only outer loop, no error stats, much faster)
    tune_on_condor: boolean, how to tune: sequentially on a machine or parallel on bdap
    
    If condor run is requested it set up a condor monitoring routine (monitor_condor_q) to follow progress
    and warn in case of jobs put on hold. This monitoring stops when there are no more jobs in the que,
    monitor_condor_q will check that all spec files used have a corresponding output file 
    """
    ##########################################################################################
    # USER PARAMS
    # be care forecstingMonths is in config!
    # month X means that all months up to X (included) are used, so this is possible in month X+1
    config_fn = upar.config_fn
    run_name = upar.run_name
    runType = upar.runType
    tune_on_condor = upar.tune_on_condor
    time_step_check = upar.time_step_check

    # the class mlSettings of a10_config sets all the possible configuration to be tested.
    # The user can reduce the numbers of possible configuration in a given run by editing
    # the function config_reducer in  a10_config

    # END OF USER PARAMS
    ##########################################################################################

    config = a10_config.read(config_fn, run_name, run_type=runType)
    forecastingMonths = config.forecastingMonths
    spec_files_list = tuner.tuneA(run_name, config_fn, tune_on_condor, runType)
    tuner.tuneB(run_name, config_fn, tune_on_condor, runType, spec_files_list)
    if tune_on_condor:
        print('Condor runs launched, start the monitoring')
        # Start the monitoring loop in a separate thread to avoid blocking the main program
        thread = threading.Thread(target=monitor_condor_q, args=(time_step_check, 'ml4castproc', config, run_name)) #60 is min to wait for checking
        thread.start()
