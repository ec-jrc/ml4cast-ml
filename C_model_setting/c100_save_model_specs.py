from pathlib import Path
import itertools
import json
import os
import numpy as np

def save_model_specs(config, modelSettings):
    """
    Save the settings of each single model to be tested
    :param config:
    :param modelSettings:
    :return: -
    """

    #### Configuration setting from config
    runID = 0
    # save files for benchmark
    Bench_algos = list(modelSettings.benchmarks)
    a = [config.crops, Bench_algos, modelSettings.forecastingMonths]
    combs = list(itertools.product(*a))
    for crop, algo, forecast_time in combs:
        # Save model settings as json
        uset = {'runID': runID,
                'crop': crop,
                'algorithm': algo,
                'dataScaling': '',
                'feature_set': 'none',
                'feature_groups': 'none',
                'feature_selection': 'none',
                'data_reduction': 'none',
                'PCAprctVar2keep': 'none',
                'prct_features2select_grid': 'none',
                # 'n_features2select_grid': 'none',
                'doOHE': 'none',
                'forecast_time': forecast_time,
                'addYieldTrend': 'none',
                'ny_max_trend': modelSettings.ny_max_trend,
                'scoringMetric': modelSettings.scoringMetric,
                'nJobsForGridSearchCv': modelSettings.nJobsForGridSearchCv}
        myID = f'{runID:06d}'
        with open(os.path.join(config.models_spec_dir, myID + '_' + crop + '_' + algo + '.json'), 'w') as fp:
            json.dump(uset, fp, indent=4)
        runID = runID + 1

    # save files for ML
    ML_algos = list(modelSettings.hyperGrid.keys())
    # repetat the models with feature eng settings
    for ften in modelSettings.ft_eng:
        # DEBUG: run only ft eng models
        # ML_algos = [x+ften for x in ML_algos]
        ML_algos = ML_algos + [x + ften for x in ML_algos]
    feature_sets = list(modelSettings.feature_groups.keys())
    a = [config.crops, ML_algos, modelSettings.forecastingMonths,
         modelSettings.doOHEs, feature_sets, modelSettings.feature_selections, modelSettings.dataReduction, modelSettings.addYieldTrend]
    combs = list(itertools.product(*a))
    # sort comb to have MRMR cases first, as they are the slowest
    combs = sorted(combs, key=lambda x: x[5])

    # And loop over
    for crop, algo, forecast_time, doOHE, feature_set, ft_sel, data_redct, addYieldTrend in combs:
        skip = False  # always false except when feature selection is requested but the length of the grid
        # of feature numbers results to be 1, meaning that it was 1 already so no feature selection possible
        # skip it if PCA is requested but we only have one month (??)
        if forecast_time == 1 and data_redct == 'PCA':
            skip = True
        if '@' in algo:
            # it is a ML model working of ft eng, we need to get the algo name to get hyper grid
            algo_name = algo.split("@")[0]
        else:
            algo_name = algo
        # Save model settings as json
        uset = {'runID': runID,
                'crop': crop,
                'algorithm': algo,
                'hyperGrid': modelSettings.hyperGrid[algo_name],
                'dataScaling': modelSettings.dataScaling,
                'feature_set': feature_set,
                'feature_groups': modelSettings.feature_groups[feature_set],
                'feature_selection': ft_sel,
                'data_reduction': data_redct,
                'PCAprctVar2keep': modelSettings.PCAprctVar2keep,
                'prct_features2select_grid': modelSettings.feature_prct_grid,
                # 'n_features2select_grid': n_features2select_grid,
                'doOHE': doOHE,
                'forecast_time': forecast_time,
                'addYieldTrend': addYieldTrend,
                'ny_max_trend': modelSettings.ny_max_trend,
                'scoringMetric': modelSettings.scoringMetric,
                'nJobsForGridSearchCv': modelSettings.nJobsForGridSearchCv}
        myID = f'{runID:06d}'

        # doRun = crop == 'Soybeans' and algo == 'GPR' and feature_set == 'rs_met_reduced' and ft_sel == 'none' and data_redct == 'none' and addYieldTrend == False
        # doRun = doRun or (crop == 'Soybeans' and algo == 'Lasso' and feature_set == 'rs_sm_reduced' and ft_sel == 'MRMR' and data_redct == 'PCA' and addYieldTrend == True)
        # doRun = doRun or (
        #             crop == 'Sunflower' and algo == 'SVR_linear' and feature_set == 'rs_reduced' and ft_sel == 'MRMR' and data_redct == 'none' and addYieldTrend == False)
        # doRun = doRun or (
        #         crop == 'Sunflower' and algo == 'Lasso' and feature_set == 'rs' and ft_sel == 'MRMR' and data_redct == 'PCA' and addYieldTrend == False)
        # doRun = doRun or (
        #         crop == 'Maize_total' and algo == 'GPR' and feature_set == 'rs_reduced' and ft_sel == 'none' and data_redct == 'none' and addYieldTrend == True)
        # doRun = doRun or (
        #         crop == 'Maize_total' and algo == 'Lasso' and feature_set == 'rs' and ft_sel == 'none' and data_redct == 'PCA' and addYieldTrend == True)
        # skip = not(doRun)
        if skip == False:
            # debug
            print(runID, crop, algo, forecast_time, doOHE, feature_set, ft_sel, data_redct, addYieldTrend)
            with open(os.path.join(config.models_spec_dir, myID + '_' + crop + '_' + algo + '.json'), 'w') as fp:
                json.dump(uset, fp, indent=4)#json.dump(uset, fp)
            runID = runID + 1

    # if not cst.is_condor:  # run locally
    #     for x in jsonList:
    #         condor_launcher.launcher(x)
    # print('End')