import os
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
import calendar
import datetime

from A_config import a10_config
from B_preprocess import b100_load
from D_modelling import d100_modeller, d110_benchmark_models
from F_post_processsing import F110_process_opeForecast_output
import manager_0_user_params as upar


if __name__ == '__main__':
    '''
    This script is used to run the operational yield forecast for a new year. Predictors are saved in a
    different dir that can be updated and to avoid overwrite of features used for training (OpeForecast_data under root dir
    specified in config file).
    The script first refit the best model pipeline on all available years and then make the forecast.
    The best model (can be also benchmark model is selected by crop and admin unit)
    '''
    pd.set_option('display.width', 5000)
    pd.set_option('display.max_columns', None)
    ##########################################################################################
    # USER PARAMS
    metric = upar.metric  # metric for best model selection, while rRMSE_ponly this implemnted for now because of graphs and maps
    config_fn = upar.config_fn
    run_name = upar.run_name
    metric_for_model_selection = upar.metric
    config_ope = a10_config.read_ope(config_fn)
    forecastingMonth = config_ope.forecastingMonth  # month X means that all months up to X (included) are used, so this is possible in month X+1
    forecastingYear = config_ope.Year  # This year refer to time of EOS
    tune_on_condor = upar.tune_on_condor
    # END OF USER PARAMS
    ##########################################################################################


    start_time = time.time()

    # load region specific data info
    config = a10_config.read(config_fn, run_name)
    # pass the year info to standard confi to reach preprocess
    config.forecastingYear = forecastingYear
    mlsettings = a10_config.mlSettings(forecastingMonths=forecastingMonth)
    config.forecastingMonth_ope = forecastingMonth
    runType = 'opeForecast'
    # get the month when forecasts are issued
    # forecastingMonths is month in the season (1,e, .. from the first), forecastingCalendarMonths is the calendar month
    di = dict(zip(config.forecastingMonths, config.forecastingCalendarMonths))
    # the forecast is issue early in the month after the last month used
    forecast_issue_calendar_month = di[forecastingMonth] + 1
    if forecast_issue_calendar_month > 12:
        forecast_issue_calendar_month = forecast_issue_calendar_month - 12
    forecast_issue_calendar_month = calendar.month_abbr[forecast_issue_calendar_month]
    # make necessary directories
    config.ope_run_dir = config.ope_run_dir + '_mInSeas' + str(forecastingMonth) + '_Y' + str(forecastingYear)
    Path(config.ope_run_dir).mkdir(parents=True, exist_ok=True)
    config.ope_run_out_dir = os.path.join(config.ope_run_dir, 'output')
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
        df_best_time_crop = df_best_time[df_best_time['Crop'] == crop]
        #get best (can ML or bench)
        df_best = df_best_time_crop.loc[df_best_time_crop[metric_for_model_selection] == df_best_time_crop[metric_for_model_selection].min()]
        list2run = mlsettings.benchmarks.copy()
        list2run.append(df_best['Estimator'].iloc[0])
        # if best is bench don't do it twice (remove duplicates from list using set)
        list2run = sorted(list(set(list2run)))
        for est in list2run:    # make forecasts with the 2 or 3 estimators left
            print(crop, est)
            df_run = df_best_time_crop.loc[df_best_time_crop['Estimator'] == est]
            if est not in mlsettings.benchmarks:
                df_run = df_run.loc[df_run[metric_for_model_selection] == df_run[metric_for_model_selection].min()]
            # get the run id
            runID = df_run['runID'].values[0]
            # get the spec of the file and build specification file
            myID = f'{runID:06d}'
            fn_spec = os.path.join(config.models_spec_dir, myID + '_' + crop + '_' + est + '.json')
            with open(fn_spec, 'r') as fp:
                uset = json.load(fp)
            print(uset)
            forecast_fn = os.path.join(config.ope_run_out_dir, datetime.datetime.today().strftime('%Y%m%d') + '_' +
                                       uset['crop'] + '_forecast_mInSeas' + str(uset['forecast_time'])
                                       + '_early_' + str(forecast_issue_calendar_month) + '_' + uset[
                                           'algorithm'] +
                                       '.csv')
            if not os.path.exists(forecast_fn):
                # set pipeline specs
                forecaster = d100_modeller.YieldModeller(uset)
                # preprocess data according to specs
                X, y, groups, feature_names, adm_ids = forecaster.preprocess(config, runType)

                # X, y, groups extend beyond the years for which I have yield data (at least one year more, the year being forecasted):
                # the years used for training (from year_start to year_end) in the config json.
                # Here I split X, y in two set, the fitting and the forecasting one.
                fit_indices = np.where(np.logical_and(groups >= config.year_start, groups <= config.year_end))[0]
                forecast_indices = np.where(groups == forecastingYear)[0]
                # fit
                # Benchmarks require special handing
                if est == 'Null_model':
                    forecasts = []
                    au_codes = []
                    for adm_id in np.unique(adm_ids):
                        # fit (i.e. compute the average)
                        ind_adm_id_in_fit_indices = np.where(adm_ids[fit_indices] == adm_id) # this is a subset of fit indices
                        forecasts.append(np.nanmean(y[fit_indices[ind_adm_id_in_fit_indices]]))
                        au_codes.append(adm_id)
                elif est == 'Trend':
                    forecasts = []
                    au_codes = []
                    for adm_id in np.unique(adm_ids):
                        # the trend is precomputed in X, I only need to get it
                        ind_adm_id_in_forecast_indices = np.where(adm_ids[forecast_indices] == adm_id)  # this is a subset of forecast indices
                        forecasts.append(np.nanmean(X[forecast_indices[ind_adm_id_in_forecast_indices]][0][0]))
                        au_codes.append(adm_id)
                elif est == 'PeakNDVI':
                    forecasts = []
                    au_codes = []
                    for adm_id in np.unique(adm_ids):
                        ind_adm_id_in_fit_indices = np.where(adm_ids[fit_indices] == adm_id)  # this is a subset of fit indices
                        ind_adm_id_in_forecast_indices = np.where(adm_ids[forecast_indices] == adm_id)  # this is a subset of forecast indices
                        y_true, y_pred, reg_list = d110_benchmark_models.run_fit(est, X[fit_indices[ind_adm_id_in_fit_indices]], y[fit_indices[ind_adm_id_in_fit_indices]], adm_ids[fit_indices[ind_adm_id_in_fit_indices]])
                        res = reg_list[0].predict(X[forecast_indices[ind_adm_id_in_forecast_indices]])
                        forecasts.append(res[0])
                        au_codes.append(adm_id)
                elif est == 'Tab':
                    # for Tab I use d110_benchmark_models.run_LOYO that fit on a sample and predict on another
                    # d110_benchmark_models.run_LOYO(model, X_train, X_test, y_train, y_test, adm_id_train, adm_id_test, groups_test)
                    y_pred, y_test, adm_id_test, year_test = d110_benchmark_models.run_LOYO(est, X[fit_indices, :], X[forecast_indices, :], y[fit_indices], y[forecast_indices],
                                                   adm_ids[fit_indices], adm_ids[forecast_indices], groups[forecast_indices])
                    au_codes = adm_id_test
                    forecasts = y_pred
                else: # It is ML
                    hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, prctPegged, \
                    selected_features_names, prct_selected, n_selected, \
                    avg_scoring_metric_on_val, fitted_model = forecaster.fit(X[fit_indices, :], y[fit_indices], groups[fit_indices], feature_names, adm_ids[fit_indices], runType)
                    # The features to be used are stored selected_features_names, extract them from X
                    ind2retain = [np.where(np.array(feature_names)==item)[0][0] for item in selected_features_names]
                    # apply the fitted model to forecast data
                    forecasts = fitted_model.predict(X[forecast_indices, :][:, np.array(ind2retain)]).tolist()
                    au_codes = adm_ids[forecast_indices].tolist()
                if mlsettings.setNegativePred2Zero == True:
                    forecasts = [x if x >= 0 else 0 for x in forecasts]

                F110_process_opeForecast_output.to_csv(config, forecast_issue_calendar_month, forecaster.uset, au_codes, forecasts, runID = runID)

    # Add forecasting year and planting year to config (for NASA format)

    # Here I make the best accuracy (by admin) estimates, and conservative estimates, and make state level estimates
    config.harvest_year = forecastingYear
    F110_process_opeForecast_output.make_consolidated_ope(config)
    print('end ope forecast')