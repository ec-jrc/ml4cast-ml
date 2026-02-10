import sys
import glob
import shutil
import os
import pandas as pd
from pathlib import Path
from F_post_processsing import F100_analyze_hindcast_output
from A_config import a10_config
import tuner
import threading
import manager_0_user_params as upar
from manager_20_tune import monitor_condor_q

if __name__ == '__main__':
    """
    Condor works with one job at at a time in scikit and has multiprocess disabled.
    In this conidtions it is fast in general but may be slow with specific run that can run forever..
    This script make a list of the slow process (thos that stay in run after some time) to be run locally 
    with more jobs and multithread. It makes a list and zip it with the spec file for local run.
    On local computer, make sure that config, data, manager_0, are all equal to those on jeo.

    """
    # USER PARAMS
    metric = upar.metric  # metric for best model selection, RMSE_val is the only one avail in fast_tuning
    n = upar.n  # ml models to rerun (obsrvation show that the best model found by standard tuning is within the first 10 found by fast tuning
    config_fn = upar.config_fn
    run_name = upar.run_name
    run_type = 'fast_tuning'
    # END OF USER PARAMS

    config = a10_config.read(config_fn, run_name,
                             run_type=run_type)  # only to get where the results were stored
    out_fast = config.models_out_dir
    F100_analyze_hindcast_output.gather_output(config)
    runIDs2rerun = F100_analyze_hindcast_output.compare_fast_outputs(config, n, metric2use=metric)
    listSlow_bn = []
    listSlow_fn = []
    config = a10_config.read(config_fn, run_name, run_type='tuning')

    # check who di not produced output
    for id in runIDs2rerun:
        myID = f'{id:06d}'
        if len(glob.glob(os.path.join(config.models_out_dir, '*' + myID + '*'))) == 0:
            bn = os.path.basename(glob.glob(os.path.join(config.models_spec_dir, '*' + myID + '*'))[0])
            fn = glob.glob(os.path.join(config.models_spec_dir, '*' + myID + '*'))[0]
            listSlow_bn.append(bn)
            listSlow_fn.append(fn)
    outDir = os.path.join(config.models_dir, 'listSlow')
    Path(outDir).mkdir(parents=True, exist_ok=True)
    for fn in listSlow_fn:
        shutil.copy(fn, outDir)
    # df = pd.DataFrame(listSlow_bn, columns=['listSlow'])
    # df.to_csv(os.path.join(outDir, 'listSlow.csv'))
    # zip all the dire and move it shared folder
    jeo_share_root = '/mnt/cidstorage/cidportal/data/cid-bulk22/Shared/tmp/projectData/ML4CAST/'
    output_filename = os.path.join(jeo_share_root, 'slowRuns')
    shutil.make_archive(output_filename, 'zip', outDir)
    print('Files copied')
