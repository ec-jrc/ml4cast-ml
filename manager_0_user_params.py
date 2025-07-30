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
    # AO Angola
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\AO\Main\Maize_(corn)_WC-Angola-harvestat.json'
    # run_name = 'AO_2025017'
    # BE Benin ASAP
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\BE\BEMaize_(corn)_WC-Benin-ASAP.json'
    # run_name = 'BE_20241226'
    # BE Benin viirs ASAP
    # config_fn = r'V:\foodsec\Projects\SNYF\SIDv\BE\BEMaize_(corn)_WC-Benin-ASAP.json'
    # run_name = 'BEv_20250709'
    # MA multi shp
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\MA\MAfews_config.json'
    # run_name = 'MA_20250331'
    # Morocco multi shp after boundaries optimization
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\MA\MAfews_config.json'
    # run_name = 'MA_20250709'
    # DZ
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\DZ\DZMultiple_WC-Algeria-ASAP_config.json'
    # run_name = 'DZ_20250131'
    # Morocco MA
    config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\MA\MAfews_config.json'
    run_name = 'MA_20250729'


    # MW Malawi
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\MW\main\Maize_(corn)_WC-Malawi-HARVESTAT.json'
    # run_name = 'MW_20250123'
    # MZ Mozambique ASAP
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\MZ\main\Maize_(corn)_WC-Mozambique-FEWSNET.json'
    # run_name = 'MZ_20250130'
    # MZ Mozambique ASAP
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\MZ\main\Maize_(corn)_WC-Mozambique-FEWSNET.json'
    # run_name = 'MZ_20250527'
    # NE Niger
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\NE\Main\NEMain_Maize_(corn)_WC-Niger-HARVESTAT_config.json'
    # run_name = 'NE_20250527'
    # run_name = 'SD_test'
    # NG Nigeria
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\NG\Wet\NGMaize_(corn)_WC-HARVESTAT_config.json'
    # run_name = 'NG_test'
    # NG Nigeria bi
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\NG\Wet_bi\NG_bi_Maize_(corn)_WC-HARVESTAT_config.json'
    # run_name = 'NGbi_20250507'
    # # NG Nigeria mono correct split
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\NG\Wet_mono\NG_mono_Maize_(corn)_WC-HARVESTAT_config.json'
    # run_name = 'NGmono_20250407'
    # SD Sudan ASAP sorghum millet
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\SD\SD_sorghum_millet.json'
    # run_name = 'SD_sor_mil_20250428'
    # SO Somalia Gu Agropastoral
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\SO\SOGuAgrop\SOGuAgrop_Somalia_Maize_rainfed-Somalia-HARVESTAT_config.json'
    # run_name = 'SOGuAgrop_20250403'
    # SO Somalia Gu Agropastoral with exclusions
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\SO\SOGuAgrop\SOGuAgrop_Somalia_Maize_rainfed-Somalia-HARVESTAT_config_exclusions.json'
    # run_name = 'SOGuAgrop_20250428excl'
    # SO Somalia Gu Riverine
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\SO\SOGuRiver\SOGuRiver_Somalia_Maize_irrigated-Somalia-HARVESTAT_config.json'
    # # SO Somalia Gu Riverine with exclusions
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\SO\SOGuRiver\SOGuRiver_Somalia_Maize_irrigated-Somalia-HARVESTAT_config_exclusions.json'
    # run_name = 'SOGuRiver_2025414excl'
    # # SO Somalia Deyr Agropastoral
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\SO\SODeyrAgrop\SODeyrAgrop_Somalia_Maize_rainfed-Somalia-HARVESTAT_config.json'
    # run_name = 'SODeyrAgrop_20250404'
    # # SO Somalia Deyr River
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\SO\SODeyrRiver\SODeyrRiver_Somalia_Maize_rainfed-Somalia-HARVESTAT_config.json'
    # run_name = 'SODeyrRiver_20250404'
    # TN
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\TN\Winter2\TNMultiple_WC-Tunisia-ASAP_config_excl.json'
    # run_name = 'TN_20250424'
    # TN viirs
    # config_fn = r'V:\foodsec\Projects\SNYF\SIDv\TN\Winter2\TNMultiple_WC-Tunisia-ASAP_config_excl.json'
    # run_name = 'TNv_20250704'
    # UA
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\UA\UA_a4c_config.json'
    # run_name = 'UA20250516'
    # UA ASAP data and afi
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\UA\UA_a4c_config_ASAP_data_ASAP_AFI.json'
    # run_name = 'UA20250519a'
    # UA viirs ASAP data and afi
    # config_fn = r'V:\foodsec\Projects\SNYF\SIDv\UA\UA_a4c_config_ASAP_data_ASAP_AFI.json'
    # run_name = 'UAv202500704'
    # UA ASAP data and afi, model tune on Modis and run on VIIRS
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\UAtest\UA_a4c_config_ASAP_data_ASAP_AFI.json'
    # run_name = 'UA20250519a'
    # ZA
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\ZA\summer\ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'
    # run_name = 'ZA_20241226'

    # ZA (test for tab)
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\ZA\\summer2024data\ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config_tab_test.json'
    # run_name = 'ZA_tab_test3'

    # ZA con 2024 data
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\ZA\summer2024data\ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'
    # run_name = 'ZA_20250410'
    # ZA viirs con 2024 data
    # config_fn = r'V:\foodsec\Projects\SNYF\SIDv\ZA\summer2024data\ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'
    # run_name = 'ZAv_20250703'
    # # ZM Zambia
    # config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\ZM\annual\Maize_(corn)_WC-Zambia-HARVESTAT.json'
    # run_name = 'ZM_20250403'




else:
    # AO Angola
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/AO/Main/Maize_(corn)_WC-Angola-harvestat.json'
    # run_name = 'AO_2025017'
    # DZ
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/DZ/DZMultiple_WC-Algeria-ASAP_config.json'
    # run_name = 'DZ_20250131'
    # BE Benin ASAP
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/BE/BEMaize_(corn)_WC-Benin-ASAP.json'
    # run_name = 'BE_20241226'
    # MA Morocco variable boundaries
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/MA/MAfews_config.json'
    # # # run_name = 'MA_20250404'
    # # MA WITHOUT GPR MA_20250512
    # run_name = 'MA_20250512'
    # MW Malawi
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/MW/main/Maize_(corn)_WC-Malawi-HARVESTAT.json'
    # run_name = 'MW_20250123'
    # MZ Mozambique
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/MZ/main/Maize_(corn)_WC-Mozambique-FEWSNET.json'
    # run_name = 'MZ_20250130'
    # MZ Mozambique shorter calendar and 90 % area
    config_fn = r'/eos/jeodpp/data/projects/ML4CAST/MZ/main/Maize_(corn)_WC-Mozambique-FEWSNET.json'
    run_name = 'MZ_20250527'
    # NG Nigeria bi correct split
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/NG/Wet_bi/NG_bi_Maize_(corn)_WC-HARVESTAT_config.json'
    # run_name = 'NGbi_20250407'
    # # NG Nigeria mono correct split
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/NG/Wet_mono/NG_mono_Maize_(corn)_WC-HARVESTAT_config.json'
    # run_name = 'NGmono_20250407'
    # Somalia
    # Gu Agropop
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/SOGuAgrop/SOGuAgrop_Somalia_Maize_rainfed-Somalia-HARVESTAT_config.json'
    # run_name = 'SOGuAgrop_20250403'
    # Gu Agropop with exclusions
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/SOGuAgrop/SOGuAgrop_Somalia_Maize_rainfed-Somalia-HARVESTAT_config_exclusions.json'
    # run_name = 'SOGuAgrop_20250428excl'
    # Gu River
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/SOGuRiver/SOGuRiver_Somalia_Maize_irrigated-Somalia-HARVESTAT_config.json'
    # run_name = 'SOGuRiver_20250404'
    # Gu River with exclusion
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/SOGuRiver/SOGuRiver_Somalia_Maize_irrigated-Somalia-HARVESTAT_config_exclusions.json'
    # run_name = 'SOGuRiver_2025414excl'
    # Dyer Agropop
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/SODeyrAgrop/SODeyrAgrop_Somalia_Maize_rainfed-Somalia-HARVESTAT_config.json'
    # run_name = 'SODeyrAgrop_20250404'
    # Dyer river
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/SODeyrRiver/SODeyrRiver_Somalia_Maize_rainfed-Somalia-HARVESTAT_config.json'
    # run_name = 'SODeyrRiver_20250404'
    # TN Tunisia
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/TN/Winter2/TNMultiple_WC-Tunisia-ASAP_config.json'
    # run_name = 'TN_20250411'
    # TN Tunisia with exclusions
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/TN/Winter2/TNMultiple_WC-Tunisia-ASAP_config_excl.json'
    # run_name = 'TN_20250424'
    # UA
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/UA/UA_a4c_config.json'
    # run_name = 'UA20250516'
    # UA with asap adata and afi
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/UA/UA_a4c_config_ASAP_data_ASAP_AFI.json'
    # run_name = 'UA20250519a'
    # SD Sudan
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/SD/SD_sorghum_millet.json'
    # run_name = 'SD_sor_mil_20250428'
    # ZA
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZA/summer/ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'
    # run_name = 'ZA_20241226'
    # ZA2 data updated to 2024
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZA/summer2024data/ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'
    # run_name = 'ZA_20250410'
    # ZM zambia
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZM/annual/Maize_(corn)_WC-Zambia-HARVESTAT.json'
    # run_name = 'ZM_20250403'
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