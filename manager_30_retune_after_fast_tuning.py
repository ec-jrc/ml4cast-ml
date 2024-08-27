import sys
import glob
import shutil
import os
from pathlib import Path
from F_post_processsing import F100_analyze_hindcast_output
from A_config import a10_config
import tuner
import threading
from manager_20_tune import monitor_condor_q


if __name__ == '__main__':
    """
    This manager collect outputs from the quick_tuning run(dir output_fast_tuning), take teh best n models according to RMSE_val,
    rerun them properly with standard tuning and place it, together with benchmarks in dir output  
    """
    # USER PARAMS
    metric = 'RMSE_val' #metric for best model selection, RMSE_val is the only one avail in fast_tuning
    n = 4 # ml models to rerun
    # env = 'pc' #['pc','jeo']
    # if env == 'pc':
    if 'win' in sys.platform:
        config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\ZA\summer\ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'  # r'V:\foodsec\Projects\SNYF\NDarfur\NDarfur_config.json'
        run_name = 'months5and7'  # 'test_quick'
        # runType = 'fast_tuning'  # 'fast_tuning'  # this is fixed for tuning ['tuning', 'fast_tuning', 'opeForecast']
        tune_on_condor = False
    else:
        config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZA/summer/ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'
        run_name = 'months5onlyMaize' #'month5and7'
        # runType = 'fast_tuning'  # this is fixed for tuning ['tuning', 'fast_tuning', 'opeForecast']
        tune_on_condor = True
    # END OF USER PARAMS


    print('Make sure that all output files were produced (as confirmed by manager_20_tune. If not rerun manager_20_tune')
    pro = input('Type Y to proceed\n')
    if pro != 'Y':
        sys.exit()
    config = a10_config.read(config_fn, run_name, run_type='fast_tuning') #only to get where the fast tuning was stored
    out_fast = config.models_out_dir
    F100_analyze_hindcast_output.gather_output(config)
    runIDs2rerun = F100_analyze_hindcast_output.compare_fast_outputs(config, n, metric2use=metric)
    # move the benchmark and run properly the ml to be rerun
    config = a10_config.read(config_fn, run_name, run_type='tuning')
    out_standard= config.models_out_dir
    Path(out_standard).mkdir(parents=True, exist_ok=True)
    # copy benchmarks (names in mlsettings.benchmarks)
    mlsettings = a10_config.mlSettings(forecastingMonths=0)
    spec_files_list = []
    for ben in mlsettings.benchmarks:
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
                                  args=(1, 'ml4castproc', config, run_name))  # 60 is min to wait for checking
        thread.start()
    print('end')


