from D_modelling import d090_model_wrapper
from A_config import a10_config

fn = r'V:\foodsec\Projects\SNYF\stable_input_data\ZA\summer\RUN_Maize_(corn)_WC-South_Africa-ASAP\TUNE_20241118\Specs\000849_Sunflower_SVR_linear@PeakFPARAndLast3.json'
config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\ZA\summer\ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'
runType = 'fast_tuning'
run_name = '20241118'
config = a10_config.read(config_fn, run_name)
d090_model_wrapper.fit_and_validate_single_model(fn, config, runType, run2get_mres_only=False)