"""
Processing
-	No double loop, we know what are best model configurations in prediction
-	We tune the selected model at time t over 2002-2018 and we predict the desired year. We also use null and peak again as benchmarks
-	We tune hyper in Cv, make forecast with given full configuration (conf + hyper),
-	then we apply to all past year to get and error measure (this is not available from paper analysis because hyper are reset in the outer loop)
-	We map the results, aggregate to national and show anomalies

New b100:
-	Read results of CV validation, get best models and their configuration
-	Tune hyper in CV using the full set of years (2002-2018)
-	Apply to current year (best model + benchmarks)
-	Apply to past years (fixed hyper): train on n-1 and test on 1 held out (everything conf and hyper is fixed). N models, N predictions, n true -> n deviations (historical performances)
-	Confidence of current forecast (stability of the model, tells you if you had enough data): Train on n combinations of n-1 year and forecast the current
-	Write hypers and prediction as output
"""

import copy
import numpy as np
import itertools
import glob
import os, time, datetime
import pandas as pd
from pathlib import Path
import pickle
import ast

import src.constants as cst
import b05_Init
import ml.modeller as modeller
from string import digits


#TODO: there is no trend and pCA in this version
def model_setup_as_pickle(filename, uset):
    with open(filename, 'wb') as handle:
        pickle.dump(uset, handle, protocol=4)
    return


def save_condor_launcher(filename, content):
    f_obj = open(filename, 'a')
    f_obj.write(content)
    f_obj.close()
    return

def nrt_model_manager(target, forecasting_times, forecast_month, current_year):
    """
    Inputs:
        target  the area of interest
    """
    pd.set_option('display.width', 5000)
    pd.set_option('display.max_columns', None)

    # Manage runID
    run_stamp = datetime.datetime.today().strftime('%Y%m%d')
    runID = 0
    myID = f'{run_stamp}_{runID:06d}'

    # Set paths
    tgt_dir = os.path.join(cst.odir, target, 'OPE_RUN')
    dirModel = os.path.join(cst.odir, target, 'Model', 'output')
    #Path(dirModel).mkdir(parents=True, exist_ok=True)
    dirOutModel = os.path.join(tgt_dir, 'Model', run_stamp+'_forecast_'+str(current_year)+'_'+str(forecast_month))
    Path(dirOutModel).mkdir(parents=True, exist_ok=True)

    # file for used by Condor for parallelization
    #condor_fn = os.path.join(tgt_dir, 'task_arguments_nrt.txt')
    #print(f'Saving task files in {condor_fn}')
    # if os.path.exists(condor_fn):
    #     os.remove(condor_fn)
    # Create directory for pickles
    # pkl_dir = os.path.join(tgt_dir, 'pkls')
    # Path(pkl_dir).mkdir(parents=True, exist_ok=True)

    # loop on time sampling type, type of crop, target y variable and forecast time
    project = b05_Init.init(target)
    yvar = 'Yield'
    crop_IDs = project['crop_IDs']
    crop_Names = project['crop_names']
    #crop_IDs = [1, 2, 3]  # all crops IDS
    #crop_Names = {1: 'Durum wheat', 2:'Soft wheat', 3:'Barley'}


    #lead_times = {'December': 1, 'January': 2, 'February': 3, 'March': 4, 'April': 5, 'May': 6, 'June': 7, 'July': 8}
    forecast_time = forecasting_times[forecast_month]
    # Time sampling
    tsampling = "M"

    input_fn = sorted(glob.glob(os.path.join(tgt_dir, '*pheno_features4scikit*.csv')), reverse=True)[0]
    print('####################################')
    print('Using input: ' + input_fn)


    df_best = pd.read_csv(os.path.join(dirModel, 'best_conf_of_all_models.csv'))
    print('####################################')
    print('Using best conf file: ' +os.path.join(dirModel, 'best_conf_of_all_models.csv'))
    print('make sure all is correct and updated')

    if target == 'Algeria': #the name was lead_time at time of Algeria's run
        df_run = df_best.loc[df_best['lead_time'] == forecast_time, :]
    else:
        df_run = df_best.loc[df_best['forecast_time'] == forecast_time, :]
    #df_run = df_run.loc[-df_best['Estimator'].isin(['Null']), :]
    for crop_id in crop_IDs:
        for i in ['best', 'Lasso', 'PeakNDVI']:
            df_run_c = df_run.loc[df_run['Crop'] == crop_Names[crop_id]]
            if i == 'best':
                df_run_crop = df_run_c.loc[df_run_c['RMSE_p'] == df_run_c['RMSE_p'].min()]
            elif i == 'Lasso':
                df_run_i = df_run_c.loc[df_run_c['Estimator'] == i]
                df_run_crop = df_run_i.loc[df_run_i['RMSE_p'] == df_run_i['RMSE_p'].min()]
            elif i == 'PeakNDVI':
                df_run_i = df_run_c.loc[df_run['Estimator'] == i]
                df_run_crop = df_run_i.loc[df_run_i['RMSE_p'] == df_run_i['RMSE_p'].min()]
            doOHE = df_run_crop['DoOHEnc'].values[0]
            algo = df_run_crop['Estimator'].values[0]
            ft_sel = df_run_crop['Ft_selection'].values[0]
            if ft_sel == 'MRMR':
                selected_features = df_run_crop['Selected_features_names_fit'].values[0]
                considered_features = df_run_crop['Features'].values[0]
            else:
                selected_features = df_run_crop['Features'].values[0]
                considered_features = selected_features
            selected_features = ast.literal_eval(selected_features)
            considered_features = ast.literal_eval(considered_features)
            if target == 'Algeria':
                data_reduction = 'None'
                AddYieldTrend = False
            else:
                data_reduction = df_run_crop['Data_reduction'].values[0]
                AddYieldTrend = df_run_crop['AddYieldTrend'].values[0]
                # if df_run_crop['AddYieldTrend'].values[0] == True:
                #     AddYieldTrend = True
            original_runID = df_run_crop['runID'].values[0]
            # # I have to get the feature_group, to be sent to modeller.YieldForecaster
            # varsList = ast.literal_eval(df_run_crop['Features'].values[0])
            # varSetDict = cst.feature_groups
            # # remove OHE
            # varsList = [x for x in varsList if
            #             not ('OHE' in x or 'YieldFromTrend' in x)]  # [x for x in varsList if not 'OHE' in x]
            # # remove numbers
            # remove_digits = str.maketrans('', '', digits)
            # varsList = [x.translate(remove_digits)[0:-1] for x in varsList]  # -2 to remove P or M"
            # # get unique
            # varsList = list(set(varsList))
            # bm_set = 'set not defined'
            # # check if PCA was activated
            # PCA_activated = False
            # if any(['_PC' in x for x in varsList]) == True:
            #     # remove _PC to allow assigning the feature set
            #     varsList = [x.replace('_PC', '') for x in varsList]
            #     PCA_activated = True
            #
            # for key in varSetDict.keys():
            #     if set(varSetDict[key]) == set(varsList):
            #         bm_set = key




            # Save model settings as pickle
            uset = {'original_runID': original_runID,
                   'runID': myID,
                    'dirOutModel': dirOutModel,
                    'target': target,
                    'cropID': crop_id,
                    'algorithm': algo,
                    'yvar': yvar,
                    'doOHE': doOHE,
                    'considered_features': considered_features,
                    'selected_features': selected_features,
                    #'feature_group'
                    'data_reduction': data_reduction,
                    'yieldTrend': AddYieldTrend,
                    'forecast_time': forecast_time,
                    'time_sampling': tsampling,
                    'input_data': input_fn}

            # pkl_fn = Path(os.path.join(pkl_dir, f'{myID}_uset.pkl'))
            # model_setup_as_pickle(pkl_fn, uset)

            # condor_content = f'{myID} {str(pkl_fn)} \n'
            # save_condor_launcher(condor_fn, condor_content)

            # if cst.is_condor is False:
            forecaster = modeller.YieldForecaster(uset['runID'],
                                                  uset['dirOutModel'],
                                                  uset['target'],
                                                  uset['cropID'],
                                                  uset['algorithm'],
                                                  uset['yvar'],
                                                  uset['doOHE'],
                                                  uset['considered_features'],
                                                  uset['selected_features'],
                                                  uset['forecast_time'],
                                                  uset['time_sampling'],
                                                  data_reduction = uset['data_reduction'],
                                                  yieldTrend = uset['yieldTrend']) #pass data_reduction and trend as optional to keep integrity of algeria ope
            #print(forecaster)
            print(uset)
            input_fn = uset['input_data']


            # Fit on all data excluding the year_out
            X, y, years, feature_names, regions = forecaster.preprocess(save_to_csv=True, ope_run=True,  ope_type='tuning', year_out=current_year) #forecasting tuning
            # forecaster uses data up to where stats are availble because it merges features with stats
            tic = time.time()
            forecaster.fit(X, y, years, regions)
            runTimeH = (time.time() - tic) / (60 * 60)
            print(f'Model fitted in {round(runTimeH, 4)} hours')

            # Now apply the fitted model to forecast data
            X_forecast, y_trash, years_trash, feature_trash, regions_forecast = forecaster.preprocess(save_to_csv=False, ope_run=True,
                                                                        ope_type='forecasting', year_out=current_year)
            y_forecast = forecaster.predict(X_forecast, regions_forecast)

            # Now get some uncertainty estimations
            y_uncert, y_mae = forecaster.predict_uncertainty(X, y, years, regions, X_forecast, regions_forecast)
            forecaster.to_csv(regions_forecast, y_forecast, y_uncert, y_mae, runID=original_runID)

            # increment runID
            runID += 1
            myID = f'{run_stamp}_{runID:06d}'
    return dirOutModel
    #print('End')

# if __name__ == '__main__':
#     current_year = 2022
#     print('**********************************************')
#     print('Trend and PCA are not yet implemented')
#     print('**********************************************')
#     res = input('Do you want to proceed anyhow? (Y/N)')
#     if res == 'Y':
#         nrt_model_manager(target='Algeria', forecast_month='May', current_year=current_year)


