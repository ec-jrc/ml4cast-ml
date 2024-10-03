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
    This manager just check that all files supposed to be rerun went well (condor sometimes has problems, and specs need to be rerun)
    """
    # USER PARAMS
    # The following two needs to be the same of magaer_30
    metric = 'RMSE_val' #metric for best model selection, RMSE_val is the only one avail in fast_tuning
    n = 20 # ml models to rerun (obsrvation show that the best model found by standard tuning is within the first 10 found by fast tuning
    # env = 'pc' #['pc','jeo']
    # if env == 'pc':
    if 'win' in sys.platform:
        config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\ZA\summer\ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'  # r'V:\foodsec\Projects\SNYF\NDarfur\NDarfur_config.json'
        run_name = '20240920_50_maize'  # 'test_quick'
        # runType = 'fast_tuning'  # 'fast_tuning'  # this is fixed for tuning ['tuning', 'fast_tuning', 'opeForecast']
        tune_on_condor = False
    else:
        config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZA/summer/ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'
        run_name = '20240911_75_maize'
        # runType = 'fast_tuning'  # this is fixed for tuning ['tuning', 'fast_tuning', 'opeForecast']
        tune_on_condor = True
    # END OF USER PARAMS



    config = a10_config.read(config_fn, run_name, run_type='fast_tuning') #only to get where the fast tuning was stored
    out_fast = config.models_out_dir
    F100_analyze_hindcast_output.gather_output(config)
    runIDs2rerun = F100_analyze_hindcast_output.compare_fast_outputs(config, n, metric2use=metric)
    config = a10_config.read(config_fn, run_name, run_type='tuning')
    out_standard = config.models_out_dir

    list_missing = []
    # check that output files are there
    for id in runIDs2rerun:
        myID4search = '*' + f'{id:06d}' + '*'
        res = glob.glob(os.path.join(out_standard, myID4search))
        if not res:
            list_missing.append(id)

    if not list_missing:
        print('All files present, proceed')
    else:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print('The following ids have no ouput, rerun manager_30')
        print(list_missing)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")






