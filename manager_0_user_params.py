import sys
import os
#############################
# 1. Config file, run type, name, and where to tune
#############################
# run_name = 'ZM_20241220'# '20241016_75_100_maize_sunflower_soybeans_NorthernCape' #'20241004_75_100_maize_sunflower_soybeans'

# runtype is overwritten when called by manager_50_ope or manager_30
runType = 'fast_tuning' #always 'fast_tuning'  when tuning   # this is fixed for tuning ['tuning', 'fast_tuning', 'opeForecast']

if 'win' in sys.platform:
    tune_on_condor = False
    # MA multi shp
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\MA\MAfews_config.json'
    # run_name = 'MA_20250331'
    # ZA
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\ZA\summer\ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'
    # run_name = 'ZA_20241226'
    # ZA con 2024 data
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\ZA\summer2024data\ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'
    # run_name = 'ZA_20250410'
    # DZ
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\DZ\DZMultiple_WC-Algeria-ASAP_config.json'
    # run_name = 'DZ_20250131'
    # Morocco MO
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\MO\MOAnnual-ASAP_config.json'
    # run_name = 'MO_test_mic'
    # ZM Zambia
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\ZM\annual\Maize_(corn)_WC-Zambia-HARVESTAT.json'
    # run_name = 'test'
    # BE Benin ASAP
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\BE\BEMaize_(corn)_WC-Benin-ASAP.json'
    # run_name = 'BE_20241226'
    # MZ Mozambique ASAP
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\MZ\main\Maize_(corn)_WC-Mozambique-FEWSNET.json'
    # run_name = 'MZ_20250130'
    # MW Malawi
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\MW\main\Maize_(corn)_WC-Malawi-HARVESTAT.json'
    # run_name = 'MW_per_476'
    # SO Somalia Gu Agropastoral
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\SO\SOGuAgrop\SOGuAgrop_Somalia_Maize_rainfed-Somalia-HARVESTAT_config.json'
    # run_name = 'SOGuAgrop_20250403'
    # SO Somalia Gu Riverine
    #config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\SO\SOGuRiver\SOGuRiver_Somalia_Maize_irrigated-Somalia-HARVESTAT_config.json'
    # SO Somalia Gu Riverine with exclusions
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\SO\SOGuRiver\SOGuRiver_Somalia_Maize_irrigated-Somalia-HARVESTAT_config_exclusions.json'
    # run_name = 'SOGuRiver_20250414'
    # # SO Somalia Deyr Agropastoral
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\SO\SODeyrAgrop\SODeyrAgrop_Somalia_Maize_rainfed-Somalia-HARVESTAT_config.json'
    # run_name = 'SODeyrAgrop_20250404'
    # # SO Somalia Deyr River
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\SO\SODeyrRiver\SODeyrRiver_Somalia_Maize_rainfed-Somalia-HARVESTAT_config.json'
    # run_name = 'SODeyrRiver_20250404'
    # Morocco
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\MO\MOAnnual-ASAP_config.json'
    # # run_name = 'MO_20250213'
    # SD Sudan ASAP sorghum millet
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\SD\SD_sorghum_millet.json'
    # run_name = 'SD_test'
    # NG Nigeria
    config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\NG\Wet\NGMaize_(corn)_WC-HARVESTAT_config.json'
    run_name = 'NG_test'
    # AO Angola
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\SD\SDSudan_XXX.json'
    # run_name = 'SD_xx'
    # TN
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\TN\Winter\TNMultiple_WC-Tunisia-ASAP_config.json'
    # run_name = 'TN_test'
else:
    # limit multithread (even setting njobs = 4, undelying libriaries were using more, see https://github.com/joblib/joblib/issues/793)
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    # ZA
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZA/summer/ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'
    # run_name = 'ZA_20241226'
    # DZ
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/DZ/DZMultiple_WC-Algeria-ASAP_config.json'
    # run_name = 'DZ_20250131'
    # ZM zambia
    config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZM/annual/Maize_(corn)_WC-Zambia-HARVESTAT.json'
    run_name = 'ZM_20250128'
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZM/annual/Maize_(corn)_WC-Zambia-HARVESTAT.json'
    # run_name = "ZM_20250128"
    # BE Benin ASAP
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/BE/BEMaize_(corn)_WC-Benin-ASAP.json'
    # run_name = 'BE_20241226'
    # MZ Mozambique
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/MZ/main/Maize_(corn)_WC-Mozambique-FEWSNET.json'
    # run_name = 'MZ_20250130'
    # MW Malawi
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/MW/main/Maize_(corn)_WC-Malawi-HARVESTAT.json'
    # run_name = 'MW_20250123'
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
# the param below are moved to the ope configuration file
# #############################
# # 3.Specification for ope run (runType = 'opeForecast')
# #############################
# # month X means that all months up to X (included) are used, so this is possible in month X+1
# forecastingMonth = 5 # 7
# # This year
# forecastingYear = 2024 #2023