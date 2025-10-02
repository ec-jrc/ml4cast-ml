import sys
import os

# runtype is overwritten when called by manager_50_ope or manager_30
runType = 'fast_tuning'  # ['tuning', 'fast_tuning', 'opeForecast']
# Minutes to wait for asking condor_q
time_step_check = 60
# metric for best model selection
metric = 'RMSE_val'
# ml models to rerun after fust tuning (obsrvation show that the best model found by standard tuning is within the first 10 found by fast tuning)
n = 20

if 'win' in sys.platform:
    tune_on_condor = False
    baseDir = os.path.normpath(r'V:\foodsec\Projects\SNYF\SIDv')
else:
    tune_on_condor = True
    baseDir = os.path.normpath('/eos/jeodpp/data/projects/ML4CAST/VIIRS')

# TN ObsAsSF
path_fromBaseDir = r'\TN\SF_test\ObsAsSF\TNMultiple_WC-Tunisia-ASAP_config_ObsAsForecast.json'
run_name = 'TNv_ObsAsForecast'


config_fn = os.path.join(baseDir, os.path.normpath(path_fromBaseDir).strip('\\'))
# print(config_fn)
