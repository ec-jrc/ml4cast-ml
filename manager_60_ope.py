import os
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import calendar

from A_config import a10_config
from B_preprocess import b100_load
from D_modelling import d100_modeller
from F_post_processsing import F110_process_opeForecast_output


if __name__ == '__main__':
    '''
    This script is used to run the operational yield forecast for a new year. Predictors are saved in a
    different dir that can be updated and to avoid overwrite of features used for training (OpeForecast_data under root dir
    specified in config file).
    The script first refit the best model pipeline on all available years and then make the forecast
    '''
    pd.set_option('display.width', 5000)
    pd.set_option('display.max_columns', None)
    ##########################################################################################
    # USER PARAMS
    forecastingMonth = 5  # month X means that all months up to X (included) are used, so this is possible in month X+1
    forecastingYear = 2024  # This year refer to time of EOS
    metric_for_model_selection = 'RMSE_p'  # 'RMSE_val' MUST BE USED

    if 'win' in sys.platform:
        config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\ZA\summer\ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'  # r'V:\foodsec\Projects\SNYF\NDarfur\NDarfur_config.json'
        run_name = '20240911_75_maize'
        tune_on_condor = False
    else:
        config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZAsummer/ZAsummer_config.json'
        run_name = 'xx'
        tune_on_condor = True
    # END OF USER PARAMS
    ##########################################################################################

    runType = 'opeForecast'
    start_time = time.time()

    # load region specific data info
    config = a10_config.read(config_fn, run_name)
    # get the month when forecasts are issued
    di = dict(zip(config.forecastingMonths, config.forecastingCalendarMonths))
    forecast_issue_calendar_month = di[forecastingMonth] + 1
    if forecast_issue_calendar_month > 12 :
        forecast_issue_calendar_month = forecast_issue_calendar_month - 12
    forecast_issue_calendar_month = calendar.month_abbr[forecast_issue_calendar_month]
    # make necessary directories
    Path(config.ope_run_dir).mkdir(parents=True, exist_ok=True)
    Path(config.ope_run_out_dir).mkdir(parents=True, exist_ok=True)
    # prepare data (as compared to fitting data, ope forecasts are updated to the time of analysis)
    b100_load.LoadPredictors_Save_Csv(config, runType)
    b100_load.build_features(config, runType)

    # Load model configuration to be tuned on all data
    output_analysis_dir = os.path.join(config.models_out_dir, 'Analysis')
    df_best = pd.read_csv(os.path.join(output_analysis_dir, 'all_model_output.csv'))
    print('####################################')
    print('Using best conf file: ' + os.path.join(output_analysis_dir, 'all_model_output.csv'))
    print('Make sure the files are correct and updated')
    df_best_time = df_best[df_best['forecast_time'] == forecastingMonth]
    # Tune best model, lasso and peakNDVI on all data
    print('####################################')
    print('Forecasting')
    for crop in config.crops:   # by crop to be forecasted (with the same AFI, the predictors are the same)
        df_best_time_crop = df_best_time[df_best['Crop'] == crop]
        #get best
        df_run = df_best_time_crop.loc[df_best_time_crop[metric_for_model_selection] == df_best_time_crop[metric_for_model_selection].min()]
        # if best is lasso or peak don't do it twice (remove duplicates from list using set)
        #list2run = sorted(list(set([df_run['Estimator'].iloc[0], 'Lasso', 'PeakNDVI']))) LASSO may not be selecte during fast tuning
        list2run = sorted(list(set([df_run['Estimator'].iloc[0], 'PeakNDVI'])))
        for est in list2run:    # make forecasts with the 2 or 3 estimators left
            print(crop, est)
            df_run = df_best_time_crop.loc[df_best_time_crop['Estimator'] == est]
            df_run = df_run.loc[df_run[metric_for_model_selection] == df_run[metric_for_model_selection].min()]
            # get the run id
            runID = df_run['runID'].values[0]
            # get the spec of the file and build specification file
            myID = f'{runID:06d}'
            fn_spec = os.path.join(config.models_spec_dir, myID + '_' + crop + '_' + est + '.json')
            with open(fn_spec, 'r') as fp:
                uset = json.load(fp)
            print(uset)
            # set pipeline specs
            forecaster = d100_modeller.YieldModeller(uset)
            # preprocess data according to specs
            X, y, groups, feature_names, AU_codes = forecaster.preprocess(config, runType)
            # X, y, groups extend beyond the years for which I have yield data (at least one year more, the year being forecasted):
            # the years used for training (from year_start to year_end) in the config json.
            # Here I split X, y in two set, the fitting and the forecasting one.
            fit_indices = np.where(np.logical_and(groups >= config.year_start, groups <= config.year_end))[0]
            forecast_indices = np.where(groups == forecastingYear)[0]
            # fit
            hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, prctPegged, \
                selected_features_names, prct_selected, n_selected, \
                avg_scoring_metric_on_val, fitted_model = forecaster.fit(X[fit_indices, :], y[fit_indices], groups[fit_indices], feature_names, AU_codes[fit_indices], runType)
            # The features to be used are stored selected_features_names, extract them from X
            ind2retain = [np.where(np.array(feature_names)==item)[0][0] for item in selected_features_names]
            # apply the fitted model to forecast data
            forecasts = fitted_model.predict(X[forecast_indices, :][:, np.array(ind2retain)]).tolist()
            au_codes = AU_codes[forecast_indices].tolist()
            F110_process_opeForecast_output.to_csv(config, forecast_issue_calendar_month, forecaster.uset, au_codes, forecasts, df_run['rMAE_p'].values[0],runID = runID)

    F110_process_opeForecast_output.make_consolidated_ope(config)
    print('end ope forecast')