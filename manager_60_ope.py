import os
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
import sys

from A_config import a10_config
from B_preprocess import b100_load
from D_modelling import d100_modeller
from F_post_processsing import F110_process_opeForecast_output


if __name__ == '__main__':
    '''
    This script shall be used to run the operational yield. Predictors are saved in a
       different dir that can be updated and to avoid overwrite of features used for training.
       The script first refit the model and then make the forecast
    '''
    pd.set_option('display.width', 5000)
    pd.set_option('display.max_columns', None)
    ##########################################################################################
    # USER PARAMS
    forecastingMonth = 6  # month X means that all months up to X (included) are used, so this is possible in month X+1
    forecastingYear = 2023  # This year refer to time of EOS
    metric_for_model_selection = 'RMSE_p'  # 'RMSE_val' MUST BE USED

    #env = 'pc'  # ['pc','jeo']
    #if env == 'pc':
    if 'win' in sys.platform:
        config_fn = r'V:\foodsec\Projects\SNYF\NDarfur\NDarfur_config.json'
        run_name = 'months56'
        tune_on_condor = False
    else:
        config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZAsummer/ZAsummer_config.json'
        run_name = ''
        tune_on_condor = True
    # END OF USER PARAMS
    ##########################################################################################

    runType = 'opeForecast'
    start_time = time.time()
    # load region specific data info
    config = a10_config.read(config_fn, run_name)
    # make necessary directories
    Path(config.ope_run_dir).mkdir(parents=True, exist_ok=True)
    Path(config.ope_run_out_dir).mkdir(parents=True, exist_ok=True)
    # prepare data
    b100_load.LoadPredictors_Save_Csv(config, runType)
    b100_load.build_features(config, runType)

    # Load model configuration to be tuned on all data
    df_best = pd.read_csv(os.path.join(config.models_out_dir, 'all_model_output.csv'))
    print('####################################')
    print('Using best conf file: ' + os.path.join(config.models_out_dir, 'all_model_output.csv'))
    print('make sure all is correct and updated')
    df_best_time = df_best[df_best['forecast_time'] == forecastingMonth]
    # Tune best model, lasso and peakNDVI on all data
    print('############################')
    print('Forecasting')
    for crop in config.crops:
        df_best_time_crop = df_best_time[df_best['Crop'] == crop]
        #get best
        df_run = df_best_time_crop.loc[df_best_time_crop[metric_for_model_selection] == df_best_time_crop[metric_for_model_selection].min()]
        # if best is lasso or peak don't do it twice (remove duplicates from list)
        list2run = sorted(list(set([df_run['Estimator'].iloc[0], 'Lasso', 'PeakNDVI'])))
        for est in list2run:
            print(crop, est)
            df_run = df_best_time_crop.loc[df_best_time_crop['Estimator'] == est]
            df_run = df_run.loc[df_run[metric_for_model_selection] == df_run[metric_for_model_selection].min()]
            # get the run id
            runID = df_run['runID'].values[0]
            # get the spec of the file
            myID = f'{runID:06d}'
            fn_spec = os.path.join(config.models_spec_dir, myID + '_' + crop + '_' + est + '.json')
            with open(fn_spec, 'r') as fp:
                uset = json.load(fp)
            print(uset)
            # set spec
            forecaster = d100_modeller.YieldModeller(uset)
            # preprocess data according to spec
            X, y, groups, feature_names, AU_codes = forecaster.preprocess(config, runType)
            # X, y, groups extend beyond the years for which I have yield data (at least one year more, the year being forecasted):
            # the years used for training (from year_start to year_end) in the config json.
            # here I split X, y in two set, the fitting and the forecasting one.
            fit_indices = np.where(np.logical_and(groups >= config.year_start, groups <= config.year_end))[0]
            forecast_indices = np.where(groups == forecastingYear)[0]
            # fit
            hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, prctPegged, \
                selected_features_names, prct_selected, n_selected, \
                avg_scoring_metric_on_val, fitted_model = forecaster.fit(X[fit_indices, :], y[fit_indices], groups[fit_indices], feature_names, AU_codes[fit_indices], runType)
            # The ft to used are stored selected_features_names, extract them from X
            ind2retain = [np.where(np.array(feature_names)==item)[0][0] for item in selected_features_names]
            # apply the fitted model to forecast data
            forecasts = fitted_model.predict(X[forecast_indices, :][:, np.array(ind2retain)]).tolist()
            au_codes = AU_codes[forecast_indices].tolist()
            F110_process_opeForecast_output.to_csv(config, forecaster.uset, au_codes, forecasts, df_run['rMAE_p'].values[0],runID = runID)

    F110_process_opeForecast_output.make_consolidated_ope(config)
    print('end ope forecast')