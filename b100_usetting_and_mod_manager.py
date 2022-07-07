"""
Prepares the list of input files for HTCondor
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import itertools
import os, datetime

import src.constants as cst
#import b200_gather_output
#import ml.modeller as modeller
from HTCondor import condor_launcher
import b05_Init


def model_setup_as_pickle(filename, uset):
    with open(filename, 'wb') as handle:
        pickle.dump(uset, handle, protocol=4)
    return


def save_condor_launcher(filename, content):
    f_obj = open(filename, 'a')
    f_obj.write(content)
    f_obj.close()
    return

def usetting_and_model_manager(target):
    """
    Inputs:
        target  the area of interest
    """
    project = b05_Init.init(target)
    # Set paths
    tgt_dir = project['output_dir']
    dirModel = os.path.join(tgt_dir, 'Model')
    Path(dirModel).mkdir(parents=True, exist_ok=True)


    # get crops
    crop_IDs = project['crop_IDs']
    # Edit here if less crops are to be covered
    # get forecasting time
    pheno_avg = pd.read_csv(tgt_dir + '/' + project['AOI'] + '_pheno_mean_used.csv')
    monthly_forecast_times = list(range(pheno_avg['Month_ID'].min(), pheno_avg['Month_ID'].max()+1))
    monthly_forecast_times.reverse()

    # Manage runID and paths
    run_stamp = datetime.datetime.today().strftime('%Y%m%d')
    runID = 0
    myID = f'{run_stamp}_{runID:06d}'
    pklsList = []   #list of pkls to be run
    dirOutModel = os.path.join(tgt_dir, 'Model', run_stamp)
    Path(dirOutModel).mkdir(parents=True, exist_ok=True)

    # file for used by Condor for parallelization
    condor_fn = os.path.join(tgt_dir, 'task_arguments.txt')

    if os.path.exists(condor_fn):
        os.remove(condor_fn)
    # Create directory for pickles
    pkl_dir = os.path.join(tgt_dir, 'pkls')
    Path(pkl_dir).mkdir(parents=True, exist_ok=True)
    if cst.is_condor == True:
        print('Run for HT condor')
        print(f'Saving task files in {condor_fn}')
        print(f'Saving task pkls in {pkl_dir}')

    # loop on time sampling type, type of crop, target y variable and forecast time

    # Time sampling
    time_samplings = ['P'] #["M"]  # ['P', 'M']
    # y variables to be predicted
    yvars = ['Yield'] # ['Yield', 'Production']
    # Admin unit IDs OHE types to be tested
    doOHEs = ['none', 'AU_level']  # ['none', 'AU_level', 'Cluster_level']
    # Feature sets
    feature_sets = list(cst.feature_groups.keys()) # this is default, take all, but Edit to specify less['rs_met', 'rs_met_reduced', 'rs', 'rs_reduced', 'met', 'met_reduced']
    # Add a yield estimated with trend feature
    addYieldTrend = [True, False]
    # Algorithms to be testes
    algos = list(cst.benchmarks) + list(cst.hyperGrid.keys()) # Defaults: test all the defined models
    algos = ['GPR1', 'GPR2']
    # Edit as below if you want less
    # algos =['Null_model', 'PeakNDVI', 'Trend'] # ['Null_model', 'PeakNDVI', 'Trend', 'Lasso', 'RandomForest', 'SVR_rbf', 'SVR_linear', 'MLP', 'GBR']
    # Feature selection
    feature_selections = ['none', 'MRMR']

    for time_sampling in time_samplings:
        # prediction times
        if time_sampling == 'M':
            forecast_times = monthly_forecast_times #[7, 6, 5, 4, 3, 2, 1]
        elif time_sampling == 'P':
            forecast_times = [3, 2, 1]  # all pheno phases
        else:
            print('Time sampling not defined')

        # Make the list of list
        a = [crop_IDs, yvars, [time_sampling], doOHEs, algos, feature_sets, feature_selections, forecast_times, addYieldTrend]
        combs = list(itertools.product(*a))
        # And loop over
        for crop_id, yvar, tsampling, doOHE, algo, feature_set, ft_sel, forecast_time, yieldTrend in combs:
            skip = False # always false except when feature selection is requested but the length of the grid
                         # of feature numbers results to be 1, meaning that it was 1 already so no fetaure selection possible
            n_features2select_grid = 0
            prct_features2select_grid = 0

            # Run special models only once regardless of doHE and feature set and feature selection
            # Conditions for running (if any is verified the model is run)
            isNotBenchmark = algo not in cst.benchmarks     #it is not a benchmark

            # it is benchmark and it is the first request for this set of options
            isFirstBenchmark = (algo in cst.benchmarks) and (doOHE == doOHEs[0]) and \
                               (feature_set == feature_sets[0]) and (ft_sel == feature_selections[0]) and (yieldTrend == addYieldTrend[0])
            if isNotBenchmark or isFirstBenchmark:
                # if it is a non benchmark and feature selection is on, compute n feature to select
                if (algo not in cst.benchmarks) and (ft_sel != 'none'):
                    n_features = len(cst.feature_groups[feature_set]) * forecast_time
                    prct_grid = cst.feature_prct_grid
                    n_features2select_grid = n_features * np.array(prct_grid) / 100
                    n_features2select_grid = np.round_(n_features2select_grid, decimals=0, out=None)
                    # keep those with at least 1 feature
                    idx2retain = [idx for idx, value in enumerate(n_features2select_grid) if value >= 1]
                    prct_features2select_grid = np.array(prct_grid)[idx2retain]
                    n_features2select_grid = n_features2select_grid[idx2retain]
                    # drop possible duplicate in n, and if the same number of ft referes to multiple %, take the largest (this explain the np.flip)
                    n_features2select_grid, idx2retain = np.unique(np.flip(np.array(n_features2select_grid)), return_index=True)
                    prct_features2select_grid = np.flip(prct_features2select_grid)[idx2retain]
                    if (len(n_features2select_grid) == 1):
                        skip = True

                # Save model settings as pickle
                uset = {'runID': myID,
                        'target': target,
                        'cropID': crop_id,
                        'algorithm': algo,
                        'yvar': yvar,
                        'feature_set': feature_set,
                        'feature_selection': ft_sel,
                        'prct_features2select_grid': prct_features2select_grid,
                        'n_features2select_grid': n_features2select_grid,
                        'doOHE': doOHE,
                        'forecast_time': forecast_time,
                        'yieldTrend': yieldTrend,
                        'time_sampling': tsampling}
                #print(uset)
                pkl_fn = Path(os.path.join(pkl_dir, f'{myID}_uset.pkl'))
                model_setup_as_pickle(pkl_fn, uset)
                if skip == False:
                    # No matter if I am on jeodpp or local, save task argument with save_condor_launcher
                    condor_content = f'{myID} {str(pkl_fn)} \n'
                    pklsList.append(pkl_fn)
                    save_condor_launcher(condor_fn, condor_content)
                    runID += 1
                    myID = f'{run_stamp}_{runID:06d}'


    print(f'{runID} jobs')
    if not cst.is_condor:  # run locally
        for x in pklsList:
            condor_launcher.launcher(x)
    print('End')

if __name__ == '__main__':
     usetting_and_model_manager(target='ZAsummer')
# if __name__ == '__main__':
#     usetting_and_model_manager(target='Mali')

# if skip == False:
#     # save run parameters as text file for condor
#     if cst.is_condor:
#         condor_content = f'{myID} {str(pkl_fn)} \n'
#         save_condor_launcher(condor_fn, condor_content)
#
#     else:  # if not on condor, run
#         # Set up class with parameters
#         forecaster = modeller.YieldHindcaster(myID, target, crop_id, algo, yvar, feature_set,
#                                               ft_sel, prct_features2select_grid, n_features2select_grid,
#                                               doOHE, forecast_time, tsampling)
#         X, y, groups, feature_names, AU_codes = forecaster.preprocess()
#
#         tic = time.time()
#         hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, nPegged, \
#         selected_features_names,  prct_selected, n_selected = forecaster.fit(X, y, groups, feature_names, AU_codes, save_output=True)
#         runTimeH = (time.time() - tic) / (60*60)
#         forecaster.validate(hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, nPegged, runTimeH, feature_names,
#                         selected_features_names,  prct_selected, n_selected, save_file=True)
# increment runID
