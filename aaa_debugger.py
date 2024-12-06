from D_modelling import d090_model_wrapper
from A_config import a10_config
import os

# remove limits
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
fn = r'V:\foodsec\Projects\SNYF\stable_input_data\DZ\RUN_Multiple_WC-Algeria-ASAP\TUNE_debug\Specs\000249_Durum_wheat_SVR_linear.json'
# fn = r'V:\foodsec\Projects\SNYF\stable_input_data\DZ\RUN_Multiple_WC-Algeria-ASAP\TUNE_debug\Specs\000863_Durum_wheat_SVR_linear@PeakFPARAndLast3.json'
config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\DZ\DZMultiple_WC-Algeria-ASAP_config.json'
runType = 'fast_tuning'
run_name = 'debug'
config = a10_config.read(config_fn, run_name, run_type=runType)
d090_model_wrapper.fit_and_validate_single_model(fn, config, runType, run2get_mres_only=False)