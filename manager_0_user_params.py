import sys
#############################
# 1. Config file, run type, name, and where to tune
#############################
run_name = 'YYY' #'20241028_75_100_maize_sunflower_soybeans'# '20241016_75_100_maize_sunflower_soybeans_NorthernCape' #'20241004_75_100_maize_sunflower_soybeans'
runType = 'fast_tuning'     # this is fixed for tuning ['tuning', 'fast_tuning', 'opeForecast']

if 'win' in sys.platform:
    config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\ZA\summer\ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'
    #runType = 'tuning'      # this is fixed for tuning ['tuning', 'fast_tuning', 'opeForecast']
    tune_on_condor = False
else:
    config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZA/summer/ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'
    tune_on_condor = True
    time_step_check = 60  # in minutes

#############################
# 2. Miscellaneous
#############################
# Percentage of admin to retain (based on area). I.e. rank by area and retain only the largest %
prct2retain = 100
# Minutes to wait for asking condor_q
time_step_check = 60
# metric for best model selection
metric = 'RMSE_val'
# ml models to rerun after fust tuning (obsrvation show that the best model found by standard tuning is within the first 10 found by fast tuning)
n = 20
# Specification for ope run (runType = 'opeForecast')
# month X means that all months up to X (included) are used, so this is possible in month X+1
forecastingMonth = 5
# This year
forecastingYear = 2024