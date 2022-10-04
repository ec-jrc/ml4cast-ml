from preprocess import b20_LoadCsv_savePickle, b60_build_features
from ope import b100_usetting_and_mod_manager_nrt_v2

target = 'ZAsummer'
forecasting_times = {'November': 1, 'December': 2, 'January': 3, 'February': 4, 'March': 5, 'April': 6, 'May': 7,}
# store the updated data in: -PY_data-ML1_data_input-Predictors_4OPE
if False:
    b20_LoadCsv_savePickle.LoadCsv_savePickle(target, 'Predictors_4OPE', ope_run=True)
    b60_build_features.build_features(target, ope_run=True)


print('**********************************************')
print('Trend and PCA are not yet implemented')
print('**********************************************')

# 2022 09 20 I am updating it to account for pca, trend and path changes
b100_usetting_and_mod_manager_nrt_v2.nrt_model_manager(target=target, forecasting_times= forecasting_times, forecast_month='November', current_year=2021)
