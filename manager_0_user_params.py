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
    baseDir = os.path.normpath(r'V:\foodsec\Projects\SNYF\SIDvs')
else:
    tune_on_condor = True
    baseDir = os.path.normpath('/eos/jeodpp/data/projects/ML4CAST/SIDvs')
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

# ZA Intercomp
# path_fromBaseDir = r'ZA\summer2025data\ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config1257.json'
# run_name = 'ZA'
# path_fromBaseDir = r'ZA\summer2025data\ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config346.json'
# run_name = 'ZA346'

# MW
# path_fromBaseDir = r'MW\main\Maize_(corn)_WC-Malawi-HARVESTAT.json'
# run_name = 'MWInt'

# ZW
# path_fromBaseDir = r'ZW\main\Maize_(corn)_WC-Zimbabwe-HARVESTAT.json'
# run_name = 'ZWInt'

# SF test (run name must contain SF (to reduce setting in config)
# path_fromBaseDir = r'ZA\summer2025data\SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config1235_SfAsForecast.json'
# run_name = 'ZA_SfAsSF'
# path_fromBaseDir = r'ZA\summer2025data\SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config1235_ObsAsForecast.json'
# run_name = 'ZA_ObsAsSF'
# path_fromBaseDir = r'ZA\summer2025data\SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config1235.json'
# run_name = 'ZA_NoSF'

# SF2 test (run name must contain SF (to reduce setting in config), try seasonal PT, P, and T. Use a group without faper ('met_sm_reduced')
# path_fromBaseDir = r'ZA\summer2025data\SF2\SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config12345.json'
# run_name = 'ZA_NoSF'
# path_fromBaseDir = r'ZA\summer2025data\SF2\SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config12345_ObsAsForecast.json'
# run_name = 'ZA_ObsAsSF'
# path_fromBaseDir = r'ZA\summer2025data\SF2\SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config12345_SfAsForecast.json'
# run_name = 'ZA_SfAsSF'

# ZA SF4 test, yield version 6
# path_fromBaseDir = r'ZA\summer2025data\SF4\SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config12345.json'
# run_name = 'ZA_NoSF'
# path_fromBaseDir = r'ZA\summer2025data\SF4\SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config12345_ObsAsForecast.json'
# run_name = 'ZA_ObsAsSF'
# path_fromBaseDir = r'ZA\summer2025data\SF4\SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config12345_SfAsForecast.json'
# run_name = 'ZA_SfAsSF'

# MW SF test, remember to set NO GPR, too slow !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# path_fromBaseDir = r'MW\main\SF\Maize_(corn)_WC-Malawi-HARVESTAT_config12345.json'
# run_name = 'MW_NoSF'
# path_fromBaseDir = r'MW\main\SF\Maize_(corn)_WC-Malawi-HARVESTAT_config12345_ObsAsForecast.json'
# run_name = 'MW_ObsAsSF'
# path_fromBaseDir = r'MW\main\SF\Maize_(corn)_WC-Malawi-HARVESTAT_config12345_SfAsForecast.json'
# run_name = 'MW_SfAsSF'

# ZW SF test
# path_fromBaseDir = r'ZW\main\SF\Maize_(corn)_WC-Zimbabwe-HARVESTAT_config123456.json'
# run_name = 'ZW_NoSF'
# path_fromBaseDir = r'ZW\main\SF\Maize_(corn)_WC-Zimbabwe-HARVESTAT_config123456_ObsAsForecast.json'
# run_name = 'ZW_ObsAsSF'
# path_fromBaseDir = r'ZW\main\SF\Maize_(corn)_WC-Zimbabwe-HARVESTAT_config123456_SfAsForecast.json'
# run_name = 'ZW_SfAsSF'

# ZA SF5 test, stoppo SF 3 mesi prima invece che 1
# ATTENZIONE: modificare riga 138 di d100 modeller
# path_fromBaseDir = r'ZA\summer2025data\SF5\SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config123.json'
# run_name = 'ZA_NoSF'
# path_fromBaseDir = r'ZA\summer2025data\SF5\SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config123_ObsAsForecast.json'
# run_name = 'ZA_ObsAsSF'
# path_fromBaseDir = r'ZA\summer2025data\SF5\SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config123_SfAsForecast.json'
# run_name = 'ZA_SfAsSF'


# ZA SF6 test, 90 % instead of 99
path_fromBaseDir = r'ZA\summer2025data\SF6_90\SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config12345.json'
run_name = 'ZA_NoSF'
# path_fromBaseDir = r'ZA\summer2025data\SF6_90\SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config12345_ObsAsForecast.json'
# run_name = 'ZA_ObsAsSF'
# path_fromBaseDir = r'ZA\summer2025data\SF6_90\SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config12345_SfAsForecast.json'
# run_name = 'ZA_SfAsSF'

############################################################################################
if 'win' in sys.platform:
    config_fn = os.path.join(baseDir, os.path.normpath(path_fromBaseDir).strip('\\'))
else:
    path_fromBaseDir = path_fromBaseDir.replace('\\', '/').lstrip('/')
    config_fn = os.path.join(baseDir, os.path.normpath(path_fromBaseDir).strip('\\'))
