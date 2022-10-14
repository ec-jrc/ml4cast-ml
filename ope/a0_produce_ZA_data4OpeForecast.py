from preprocess import b20_LoadCsv_savePickle, b60_build_features
from ope import b100_usetting_and_mod_manager_nrt_v2,b200_gather_outputs_nrt
import src.constants as cst

target = 'ZAsummer'
forecast_month = 'November' #'February'
forcast_year = 2022 #thi is the harvest year"

forecasting_times = {'November': 1, 'December': 2, 'January': 3, 'February': 4, 'March': 5, 'April': 6, 'May': 7,}
# store the updated data in: -PY_data-ML1_data_input-Predictors_4OPE
if False:
    b20_LoadCsv_savePickle.LoadCsv_savePickle(target, 'Predictors_4OPE', ope_run=True)
    b60_build_features.build_features(target, ope_run=True)

if True:
    for fm in forecasting_times.keys():
        print(fm)
        dirOutModel = b100_usetting_and_mod_manager_nrt_v2.nrt_model_manager(target=target, forecasting_times= forecasting_times, forecast_month=fm, current_year=forcast_year)
        b200_gather_outputs_nrt.main(target=target, folder=dirOutModel)
else:
    dirOutModel = b100_usetting_and_mod_manager_nrt_v2.nrt_model_manager(target=target, forecasting_times= forecasting_times, forecast_month=forecast_month, current_year=forcast_year)
    b200_gather_outputs_nrt.main(target=target, folder=dirOutModel)
