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
# fn = r'V:\foodsec\Projects\SNYF\stable_input_data\MW\main\RUN_Maize_(corn)_WC-Malawi-HARVESTAT\TUNE_MW_per_476\Specs\000476_Maize_GPR.json'
# fn = r'V:\foodsec\Projects\SNYF\stable_input_data\DZ\RUN_Multiple_WC-Algeria-ASAP\TUNE_debug\Specs\000863_Durum_wheat_SVR_linear@PeakFPARAndLast3.json'
# fn = r'V:\foodsec\Projects\SNYF\stable_input_data\MA\RUN_Multiple_WC-Marocco-custom_units\TUNE_MA_20250331\Specs\000000_Durum_wheat_Null_model.json'
# config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\MW\main\Maize_(corn)_WC-Malawi-HARVESTAT.json'
# runType = 'tuning'

# fn = r'V:\foodsec\Projects\SNYF\stable_input_data\SO\SOGuRiver\RUN_Somalia_SWALIM_Maize_irrigated-Somalia-HARVESTAT\TUNE_SOGuRiver_20250414\Specs\000572_Sorghum_riverine_Lasso.json'
# config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\SO\SOGuRiver\SOGuRiver_Somalia_Maize_irrigated-Somalia-HARVESTAT_config_exclusions.json'
# runType = 'fast_tuning'
# run_name = 'SOGuRiver_20250414'

# fn = r'V:\foodsec\Projects\SNYF\stable_input_data\SD\RUN_All_crops-Sudan-ASAP\TUNE_SD_test\Specs\003070_Millet_SVR_linear@PeakFPARAndLast3.json'
# config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\SD\SD_sorghum_millet.json'
# runType = 'fast_tuning'
# run_name = 'SD_test'

# fn = r'V:\foodsec\Projects\SNYF\stable_input_data\MZ\main\RUN_Maize_(corn)_WC-Mozambique-FEWSNET\TUNE_MZ_20250527\Specs\000108_Maize_SVR_linear.json'
# config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\MZ\main\Maize_(corn)_WC-Mozambique-FEWSNET.json'
# runType = 'tuning'
# run_name = 'MZ_20250527'

fn = r'V:\foodsec\Projects\SNYF\stable_input_data\MA\RUN_Multiple_WC-Marocco-custom_units\TUNE_MA_20250512\Specs\000000_Wheat_durum_Null_model.json'
config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\MA\MAfews_config.json'
runType = 'tuning'
run_name = 'MA_20250512'

config = a10_config.read(config_fn, run_name, run_type=runType)
config.nJobsForGridSearchCv = 8
d090_model_wrapper.fit_and_validate_single_model(fn, config, runType, run2get_mres_only=True)
# d090_model_wrapper.fit_and_validate_single_model(fn, config, runType, run2get_mres_only=False)