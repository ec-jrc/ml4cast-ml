from D_modelling import d090_model_wrapper
from A_config import a10_config
import os

# limit multithreat (even setting njobs = 4, undelying libriaries were using more, see https://github.com/joblib/joblib/issues/793)
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
fn = r'V:\foodsec\Projects\SNYF\stable_input_data\DZ\RUN_Multiple_WC-Algeria-ASAP\TUNE_debug\Specs\000483_Durum_wheat_GPR.json'
config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\DZ\DZMultiple_WC-Algeria-ASAP_config.json'
runType = 'fast_tuning'
run_name = 'debug'
config = a10_config.read(config_fn, run_name)
d090_model_wrapper.fit_and_validate_single_model(fn, config, runType, run2get_mres_only=False)