import os
from pathlib import Path
import shutil
import glob
import time

from A_config import a10_config
from B_preprocess import b100_load
from D_modelling import d090_model_wrapper



if __name__ == '__main__':
    # USER PARAMS
    config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\DZ\DZMultiple_WC-Algeria-ASAP_config.json'
    fn_slowRunsZip = r'V:\foodsec\Projects\SNYF\stable_input_data\DZ\RUN_Multiple_WC-Algeria-ASAP\slowRuns.zip'
    # END OF USER PARAMS



    # remove limits
    # os.environ['MKL_NUM_THREADS'] = '8'
    # os.environ['OPENBLAS_NUM_THREADS'] = '8'
    # os.environ['OMP_NUM_THREADS'] = '8'
    run_name = 'SlowRuns'
    runType = 'tuning'
    # note: by reading config on a win system, the number of jobs is set to 8
    config = a10_config.read(config_fn, run_name, run_type=runType)
    # make necessary directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.models_dir).mkdir(parents=True, exist_ok=True)
    Path(config.models_spec_dir).mkdir(parents=True, exist_ok=True)
    Path(config.models_out_dir).mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(fn_slowRunsZip, config.models_spec_dir)
    listFiles = glob.glob(os.path.join(config.models_spec_dir, '*.json'))

    # Prepare input files
    b100_load.LoadPredictors_Save_Csv(config, runType)
    b100_load.build_features(config, runType)

    for fn in listFiles:
        print(fn)
        tic = time.time()
        d090_model_wrapper.fit_and_validate_single_model(fn, config, runType, run2get_mres_only=False)
        runTimeH = (time.time() - tic) / (60 * 60)
        print('Run took ' + str(runTimeH) + 'hours')