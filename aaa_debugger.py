from D_modelling import d090_model_wrapper
from A_config import a10_config
import os
import pandas as pd

pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', None)

# remove limits
# os.environ['MKL_NUM_THREADS'] = '8'
# os.environ['OPENBLAS_NUM_THREADS'] = '8'
# os.environ['OMP_NUM_THREADS'] = '8'
fn = r'V:\foodsec\Projects\SNYF\stable_input_data\MW\main\RUN_Maize_(corn)_WC-Malawi-HARVESTAT\TUNE_MW_per_476\Specs\000476_Maize_GPR.json'
# fn = r'V:\foodsec\Projects\SNYF\stable_input_data\DZ\RUN_Multiple_WC-Algeria-ASAP\TUNE_debug\Specs\000863_Durum_wheat_SVR_linear@PeakFPARAndLast3.json'
config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\MW\main\Maize_(corn)_WC-Malawi-HARVESTAT.json'
runType = 'tuning'
run_name = 'MW_per_476'
config = a10_config.read(config_fn, run_name, run_type=runType)
config.nJobsForGridSearchCv = 8
d090_model_wrapper.fit_and_validate_single_model(fn, config, runType, run2get_mres_only=False)