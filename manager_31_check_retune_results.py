import sys
import glob
import os
from F_post_processsing import F100_analyze_hindcast_output
from A_config import a10_config
import manager_0_user_params as upar



if __name__ == '__main__':
    """
    This manager just check that all files supposed to be rerun went well (condor sometimes has problems, and specs need to be rerun)
    """
    # USER PARAMS
    metric = upar.metric  # metric for best model selection, RMSE_val is the only one avail in fast_tuning
    n = upar.n  # ml models to rerun (obsrvation show that the best model found by standard tuning is within the first 10 found by fast tuning
    config_fn = upar.config_fn
    run_name = upar.run_name
    tune_on_condor = upar.tune_on_condor
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






