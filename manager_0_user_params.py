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
############################################################################################
# TN NoSF
# path_fromBaseDir = r'\TN\SF\NO_SF_baseline\TNMultiple_WC-Tunisia-ASAP_config.json'
# run_name = 'TNv_NoSF'
# TN ObsAsSF
# path_fromBaseDir = r'\TN\SF\ObsAsSF\TNMultiple_WC-Tunisia-ASAP_config_ObsAsForecast.json'
# run_name = 'TNv_ObsAsSF'
# TN SfAsSF
# path_fromBaseDir = r'TN\SF\SF\TNMultiple_WC-Tunisia-ASAP_config_SfAsForecast.json'
# run_name = 'TNv_SfAsSF'

# Morocco
# path_fromBaseDir = r'V:\foodsec\Projects\SNYF\SIDv\MA\MAfews_config.json'
# run_name = 'MA_test'

# ZA
path_fromBaseDir = r'V:\foodsec\Projects\SNYF\SIDv\ZAInt\summer2025data\ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'
run_name = 'ZA_Int_test2'

############################################################################################
if 'win' in sys.platform:
    config_fn = os.path.join(baseDir, os.path.normpath(path_fromBaseDir).strip('\\'))
else:
    path_fromBaseDir = path_fromBaseDir.replace('\\', '/').lstrip('/')
    config_fn = os.path.join(baseDir, os.path.normpath(path_fromBaseDir).strip('\\'))
