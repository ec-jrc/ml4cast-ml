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


# fn
#
# = r'V:\foodsec\Projects\SNYF\stable_input_data\MZ\main\RUN_Maize_(corn)_WC-Mozambique-FEWSNET\TUNE_MZ_20250527\Specs\000108_Maize_SVR_linear.json'
# config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\MZ\main\Maize_(corn)_
# WC-Mozambique-FEWSNET.json'
# runType = 'fast_tuning' #'fast_tuning'
# run_name = 'MZ_20250527'

config_fn = r'V:\foodsec\Projects\SNYF\SIDvs\MW\main\Maize_(corn)_WC-Malawi-HARVESTAT.json'
run_name = 'MWInt'
fn = r'V:\foodsec\Projects\SNYF\SIDvs\MW\main\RUN_Maize_(corn)_WC-Malawi-HARVESTAT\TUNE_MWInt\Specs\000642_Maize_GPR.json'
# 000041_Ble_tendre_Lasso.json
runType = 'tuning'

# fn = r'V:\foodsec\Projects\SNYF\stable_input_data\ZA\summer2024data\RUN_Maize_(corn)_WC-South_Africa-ASAP\TUNE_ZA_20250410\Specs\000147_Sunflower_SVR_linear.json'
# config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\ZA\summer2024data\ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'
# runType = 'opeForecast'
# run_name = 'ZA_20250410'

config = a10_config.read(config_fn, run_name, run_type=runType)
config.nJobsForGridSearchCv = 8
d090_model_wrapper.fit_and_validate_single_model(fn, config, runType, run2get_mres_only=False)
# d090_model_wrapper.fit_and_validate_single_model(fn, config, runType, run2get_mres_only=False)