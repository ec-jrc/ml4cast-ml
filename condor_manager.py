import tuner
from A_config import a10_config
import datetime
import time
import threading
import subprocess
import pandas as pd

def monitor_condor_q(time_step_minutes, submitter, batch_name):
  start_time = time.time()
  first_check = True
  while True:
    current_time = datetime.datetime.now().time()


    # do something
    # JobStatus is an integer;
    # states: - 1: Idle(I) - 2: Running(R) - 3: Removed(X) - 4: Completed(C) - 5: Held(H) - 6: Transferring
    # Output - 7: Suspended
    # sudo -u ml4castproc condor_q submitter ml4castproc -format '%s ' JobBatchName -format '%-3d ' ProcId -format '%-3d\n' JobStatus
    # check_cmd = ['sudo', '-u', submitter, 'condor_q', 'submitter', submitter, '-format', "'%s '", 'JobBatchName', '-format', "'%-3d '", 'ProcId', '-format', r"'%-3d\n'", 'JobStatus']
    check_cmd = ['sudo', '-u', submitter, 'condor_q', 'submitter', submitter, '-format', "%s ", 'JobBatchName',
                 '-format', "%-3d ", 'ProcId', '-format', "%s ", 'GlobalJobId','-format', r"%-3d\n", 'JobStatus']

    print('***')
    print (" ".join(check_cmd))
    print('***')
    p = subprocess.run(check_cmd, shell=False, input='\n', capture_output=True, text=True)
    if p.returncode != 0:
        print('ERR', p.stderr)
        raise Exception('Step subprocess error')
    # print(p.stdout)
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
        print('nothing on Condor anymore, the monitoring will stop')
        print("--- %s Hours ---" % (time.time() - start_time)/(60*60))
        break
    print('Condor stats on ' + str(datetime.datetime.now()))
    if first_check:
        first_check = False
        jobRequested = len(df)
    else:
        print('Jobs sbmitted: ' + str(jobRequested))
    print('Jobs in que: ' + str(len(df)))
    print('Jobs running: ' + str(len(df[df['StatusString']=='Running'])))
    print('Jobs idle: ' + str(len(df[df['StatusString'] == 'Idle'])))
    print('Jobs held: ' + str(len(df[df['StatusString'] == 'Held'])))
    if len(df[df['StatusString'] == 'Held']) > 0:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('JOBS HELD')
        print(df[df['StatusString'] == 'Held'])
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # print(df)
    # print('here')
    # Sleep for n minutes
    time.sleep(time_step_minutes*60)


if __name__ == '__main__':
    run_name = 'month5'
    config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZAsummer/ZAsummer_config.json'
    forecastingMonths = [5]
    tune_on_condor = True
    config = a10_config.read(config_fn, run_name)
    batch_name = 'ml4cast_' + config.AOI
    tuner.tune(run_name, config_fn, forecastingMonths, tune_on_condor)
    print('Condor runs launched, start the monitoring')
    # Start the monitoring loop in a separate thread to avoid blocking the main program
    thread = threading.Thread(target=monitor_condor_q, args=(30, 'ml4castproc', batch_name))
    thread.start()
