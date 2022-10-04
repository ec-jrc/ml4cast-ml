from preprocess import b20_LoadCsv_savePickle, b60_build_features
from ope import b100_usetting_and_mod_manager_nrt_v2, b200_gather_outputs_nrt
import src.constants as cst

target = 'Algeria'
forecasting_times = {'December': 1, 'January': 2, 'February': 3, 'March': 4, 'April': 5, 'May': 6, 'June': 7, 'July': 8}
# store the updated data in: -PY_data-ML1_data_input-Predictors_4OPE
if False:
    b20_LoadCsv_savePickle.LoadCsv_savePickle(target, 'Predictors_4OPE', ope_run=True)
    b60_build_features.build_features(target, ope_run=True)

print('**********************************************')
print('Trend and PCA are not yet implemented')
print('**********************************************')

# 2022 09 20 I am updating it to account for pca, trend and path changes
b100_usetting_and_mod_manager_nrt_v2.nrt_model_manager(target=target, forecasting_times= forecasting_times, forecast_month='May', current_year=2022)
b200_gather_outputs_nrt.main(target='Algeria', folder=cst.folder_b200_gather_outputs_nrt)
