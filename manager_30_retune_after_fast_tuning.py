import sys
import glob
import shutil
import os
from pathlib import Path
from F_post_processsing import F100_analyze_hindcast_output
from A_config import a10_config
import tuner
import threading
import manager_0_user_params as upar
from manager_20_tune import monitor_condor_q


if __name__ == '__main__':
    """
    This manager collect outputs from the quick_tuning run(dir output_fast_tuning), take teh best n models according to RMSE_val,
    rerun them properly with standard tuning and place it, together with benchmarks in dir output  
    """
    # USER PARAMS
    metric = upar.metric                #metric for best model selection, RMSE_val is the only one avail in fast_tuning
    n = upar.n                          # ml models to rerun (obsrvation show that the best model found by standard tuning is within the first 10 found by fast tuning
    config_fn = upar.config_fn
    run_name = upar.run_name
    tune_on_condor = upar.tune_on_condor
    # END OF USER PARAMS



    config = a10_config.read(config_fn, run_name, run_type='fast_tuning') #only to get where the fast tuning was stored
    out_fast = config.models_out_dir
    F100_analyze_hindcast_output.gather_output(config)
    pro = input('Type Y to proceed\n')
    if pro != 'Y':
        sys.exit()
    runIDs2rerun = F100_analyze_hindcast_output.compare_fast_outputs(config, n, metric2use=metric)
    # move the benchmark and run properly the ml to be rerun
    config = a10_config.read(config_fn, run_name, run_type='tuning')
    out_standard= config.models_out_dir
    Path(out_standard).mkdir(parents=True, exist_ok=True)
    # copy benchmarks (names in mlsettings.benchmarks)
    mlsettings = a10_config.mlSettings(forecastingMonths=0)
    spec_files_list = []
    # Tab change 2025, removed, Tab is run in fast tuning where it also produces a mRes output
    # ben2copy = list(filter(lambda x: x != "Tab", mlsettings.benchmarks))
    # Tab does not need a double loop, can be copied fom output_fast_tuning to output as the other benchmarks
    # fix that now it does not write full stats

    for ben in mlsettings.benchmarks:
    # for ben in ben2copy:
        spec_files_list.extend(glob.glob(os.path.join(out_fast, '*'+ben+'*')))
    for filename in spec_files_list:
        if os.path.isfile(filename):
            shutil.copy(filename, out_standard)
    # rerun the needed models
    # get full path of spec names to be rerun
    spec_files_list = []
    for id in runIDs2rerun:
        myID = f'{id:06d}'
        spec_files_list.extend(glob.glob(os.path.join(config.models_spec_dir, myID + '*')))
    tuner.tuneB(run_name, config_fn, tune_on_condor, 'tuning', spec_files_list)
    if tune_on_condor:
        print('Condor runs launched, start the monitoring')
        # Start the monitoring loop in a separate thread to avoid blocking the main program
        thread = threading.Thread(target=monitor_condor_q,
                                  args=(10, 'ml4castproc', config, run_name))  # 60 is min to wait for checking
        thread.start()
    print('end')


