import sys
import os
#############################
# 1. Config file, run type, name, and where to tune
#############################
run_name = '20241213'# '20241016_75_100_maize_sunflower_soybeans_NorthernCape' #'20241004_75_100_maize_sunflower_soybeans'
# runtype is overwritten when called by manager_50_ope
runType = 'fast_tuning' #'fast_tuning'     # this is fixed for tuning ['tuning', 'fast_tuning', 'opeForecast']

if 'win' in sys.platform:
    # ZA
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\ZA\summer\ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'
    # DZ
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\DZ\DZMultiple_WC-Algeria-ASAP_config.json'
    # ZM Zambia
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\ZM\annual\Maize_(corn)_WC-Zambia-FEWSNET.json'
    # BE Benin ASAP
    config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\BE\BEMaize_(corn)_WC-Benin-ASAP.json'
    tune_on_condor = False
else:
    # limit multithreat (even setting njobs = 4, undelying libriaries were using more, see https://github.com/joblib/joblib/issues/793)
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    # ZA
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZA/summer/ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'
    # DZ
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/DZ/DZMultiple_WC-Algeria-ASAP_config.json'
    # ZM zambia
    config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZM/annual/Maize_(corn)_WC-Zambia-FEWSNET.json'
    tune_on_condor = True
    time_step_check = 60  # in minutes

#############################
# 2. Params for tuning
#############################
# # Percentage of admin to retain (based on area). I.e. rank by area and retain only the largest %
# prct2retain = 100 -->moved to config
# Minutes to wait for asking condor_q
time_step_check = 60
# metric for best model selection
metric = 'RMSE_val'
# ml models to rerun after fust tuning (obsrvation show that the best model found by standard tuning is within the first 10 found by fast tuning)
n = 20
#############################
# 3.Specification for ope run (runType = 'opeForecast')
#############################
# month X means that all months up to X (included) are used, so this is possible in month X+1
forecastingMonth = 5
# This year
forecastingYear = 2024