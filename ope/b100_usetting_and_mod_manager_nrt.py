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
import ml.modeller as modeller

def model_setup_as_pickle(filename, uset):
    with open(filename, 'wb') as handle:
        pickle.dump(uset, handle, protocol=4)
    return


def save_condor_launcher(filename, content):
    f_obj = open(filename, 'a')
    f_obj.write(content)
    f_obj.close()
    return

def nrt_model_manager(target, forecast_month, current_year):
    """
    Inputs:
        target  the area of interest
    """
    # Manage runID
    run_stamp = datetime.datetime.today().strftime('%Y%m%d')
    runID = 0
    myID = f'{run_stamp}_{runID:06d}'

    # Set paths
    tgt_dir = os.path.join(cst.odir, target)
    dirModel = os.path.join(tgt_dir, 'Model')
    Path(dirModel).mkdir(parents=True, exist_ok=True)
    dirOutModel = os.path.join(tgt_dir, 'Model', run_stamp)
    Path(dirOutModel).mkdir(parents=True, exist_ok=True)

    # file for used by Condor for parallelization
    condor_fn = os.path.join(tgt_dir, 'task_arguments_nrt.txt')
    print(f'Saving task files in {condor_fn}')
    if os.path.exists(condor_fn):
        os.remove(condor_fn)
    # Create directory for pickles
    pkl_dir = os.path.join(tgt_dir, 'pkls')
    Path(pkl_dir).mkdir(parents=True, exist_ok=True)

    # loop on time sampling type, type of crop, target y variable and forecast time
    yvar = 'Yield'
    crop_IDs = [1, 2, 3]  # all crops IDS
    crop_Names = {1: 'Durum wheat', 2:'Soft wheat', 3:'Barley'}
    lead_times = {'December': 1, 'January': 2, 'February': 3, 'March': 4, 'April': 5, 'May': 6, 'June': 7, 'July': 8}
    lead_time = lead_times[forecast_month]
    # Time sampling
    tsampling = "M"

    input_fn = sorted(glob.glob(os.path.join(tgt_dir, '*pheno_features4scikit*.csv')), reverse=True)[0]
    print('####################################')
    print('Using input:' + input_fn)
    print('make sure it is correct and updated')

    df_best = pd.read_csv(os.path.join(tgt_dir, 'best_conf_of_all_models.csv'))
    print('####################################')
    print('Using best conf file:' + os.path.join(tgt_dir, 'best_conf_of_all_models.csv'))
    print('make sure it is correct and updated')

    df_run = df_best.loc[df_best['lead_time'] == lead_time, :]
    df_run = df_run.loc[-df_best['Estimator'].isin(['Null']), :]
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
            else:
                selected_features = df_run_crop['Features'].values[0]
            selected_features = ast.literal_eval(selected_features)

            # Save model settings as pickle
            uset = {'runID': myID,
                    'target': target,
                    'cropID': crop_id,
                    'algorithm': algo,
                    'yvar': yvar,
                    'doOHE': doOHE,
                    'selected_features': selected_features,
                    'lead_time': lead_time,
                    'time_sampling': tsampling,
                    'input_data': input_fn}

            pkl_fn = Path(os.path.join(pkl_dir, f'{myID}_uset.pkl'))
            model_setup_as_pickle(pkl_fn, uset)

            condor_content = f'{myID} {str(pkl_fn)} \n'
            save_condor_launcher(condor_fn, condor_content)

            if cst.is_condor is False:
                forecaster = modeller.YieldForecaster(uset['runID'],
                                                      uset['target'],
                                                      uset['cropID'],
                                                      uset['algorithm'],
                                                      uset['yvar'],
                                                      uset['doOHE'],
                                                      uset['selected_features'],
                                                      uset['lead_time'],
                                                      uset['time_sampling'])
                print(forecaster)
                input_fn = uset['input_data']

                X, y, years, regions = forecaster.preprocess()
                tic = time.time()
                # forecaster uses data up to where stats are availble because it merges features with stats
                forecaster.fit(X, y, years, regions)
                runTimeH = (time.time() - tic) / (60 * 60)
                print(f'Model fitted in {round(runTimeH, 4)} hours')
                X_forecast, regions_forecast = forecaster.preprocess_currentyear(input_fn, current_year)
                y_forecast = forecaster.predict(X_forecast, regions_forecast)
                y_uncert, y_mae = forecaster.predict_uncertainty(X, y, years, regions, X_forecast, regions_forecast)
                forecaster.to_csv(regions_forecast, y_forecast, y_uncert, y_mae)

            # increment runID
            runID += 1
            myID = f'{run_stamp}_{runID:06d}'
    print('End')

if __name__ == '__main__':
    current_year = 2022
    nrt_model_manager(target='Algeria', forecast_month='May', current_year=current_year)


