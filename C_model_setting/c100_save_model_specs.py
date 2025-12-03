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
    # Use of SG
    if config.useSF == True:
        useSF = True
        #condider two options: monthly values and seasonal aggergation (eos-1 month)
        #aggregationSFs = ['monthly', 'seasonal']
        aggregationSFs = ['seasonal']
    else:
        useSF = False
        aggregationSFs = ['none']

    # save files for benchmark
    Bench_algos = list(modelSettings.benchmarks)
    a = [config.crops, Bench_algos, modelSettings.forecastingMonths, aggregationSFs]
    combs = list(itertools.product(*a))
    for crop, algo, forecast_time, aggregationSF in combs:
        skip = False
        # Tab is run only with OHE
        if algo == 'Tab':
            oheTxt = 'AU_level'
            useSFtxt = useSF
            aggregationSFtxt = aggregationSF
            addYieldTrendtXT = True
        else:
            oheTxt = 'none'
            useSFtxt = False
            aggregationSFtxt = 'none'
            addYieldTrendtXT = 'none'
        # if (algo != 'Tab') and (aggregationSF == 'seasonal') :
        #     skip = True
        # Save model settings as json
        uset = {'runID': runID,
                'crop': crop,
                'algorithm': algo,
                'dataScaling': '',
                'feature_set': 'none',
                'feature_groups': 'none',
                'useSF': useSFtxt,
                'aggregationSF': aggregationSFtxt,
                'feature_selection': 'none',
                'data_reduction': 'none',
                'PCAprctVar2keep': 'none',
                'prct_features2select_grid': 'none',
                # 'n_features2select_grid': 'none',
                'doOHE': oheTxt,
                'forecast_time': forecast_time,
                'addYieldTrend': addYieldTrendtXT,
                'ny_max_trend': modelSettings.ny_max_trend,
                'scoringMetric': modelSettings.scoringMetric,
                'nJobsForGridSearchCv': modelSettings.nJobsForGridSearchCv}
        if skip == False:
            print(runID, crop, algo, forecast_time, oheTxt, addYieldTrendtXT, useSF, aggregationSFtxt)
            myID = f'{runID:06d}'
            with open(os.path.join(config.models_spec_dir, myID + '_' + crop + '_' + algo + '.json'), 'w') as fp:
                json.dump(uset, fp, indent=4)
            runID = runID + 1

    # save files for ML
    ML_algos = list(modelSettings.hyperGrid.keys())
    # repetat the models with feature eng settings
    if modelSettings.ft_eng != None:
        for ften in modelSettings.ft_eng:
            # DEBUG: run only ft eng models
            # ML_algos = [x+ften for x in ML_algos]
            ML_algos = ML_algos + [x + ften for x in ML_algos]
    feature_sets = list(modelSettings.feature_groups.keys())


    # a = [config.crops, ML_algos, modelSettings.forecastingMonths,
    #      modelSettings.doOHEs, feature_sets, modelSettings.feature_selections, modelSettings.dataReduction, modelSettings.addYieldTrend]
    a = [config.crops, ML_algos, modelSettings.forecastingMonths,
         modelSettings.doOHEs, feature_sets, modelSettings.feature_selections, modelSettings.dataReduction,
         modelSettings.addYieldTrend, aggregationSFs]
    combs = list(itertools.product(*a))
    # sort comb to have MRMR cases first, as they are the slowest
    combs = sorted(combs, key=lambda x: x[5])

    # And loop over
    for crop, algo, forecast_time, doOHE, feature_set, ft_sel, data_redct, addYieldTrend, aggregationSF in combs:
        skip = False  # always false except when feature selection is requested but the length of the grid
        feature_group = modelSettings.feature_groups[feature_set]
        # of feature numbers results to be 1, meaning that it was 1 already so no feature selection possible
        # skip it if PCA is requested but we only have one month (??)
        if forecast_time == 1 and data_redct == 'PCA':
            skip = True
        if '@' in algo:
            # it is a ML model working of ft eng, we need to get the algo name to get hyper grid
            algo_name = algo.split("@")[0]
            if feature_set == 'met' or feature_set == 'met_reduced' or feature_set =='met_sm_reduced':
                skip = True
            if skip == False:
                # change feature group and feature set
                rad_var = modelSettings.rad_var
                bio_var = modelSettings.bio_var
                tmp = feature_set
                if tmp == 'rs_met':
                    feature_set = 'maxRS_met'
                    feature_group = [bio_var + 'max', rad_var, 'RainSum', 'T', 'Tmin', 'Tmax']
                elif tmp == 'rs_met_reduced':
                    feature_set = 'maxRS_met_reduced'
                    feature_group = [bio_var + 'max', 'RainSum', 'T']
                elif tmp == 'rs_met_sm_reduced':
                    feature_set = 'maxRS_met_sm_reduced'
                    feature_group = [bio_var + 'max', 'RainSum', 'T', 'SM']
                elif tmp == 'rs':
                    feature_set = 'maxRS'
                    feature_group = [bio_var + 'max']
                elif tmp == 'rs_reduced':
                    skip = True
                elif tmp == 'rs_sm_reduced':
                    feature_set = 'maxRS_sm'
                    feature_group = [bio_var + 'max', 'SM']
                else:
                    print('c100 feature set not defined')
                    exit()
        else:
            algo_name = algo

        # Save model settings as json
        uset = {'runID': runID,
                'crop': crop,
                'algorithm': algo,
                'hyperGrid': modelSettings.hyperGrid[algo_name],
                'dataScaling': modelSettings.dataScaling,
                'feature_set': feature_set,
                'feature_groups': feature_group,
                'useSF': useSF,
                'aggregationSF': aggregationSF,
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

        if ft_sel == 'MRMR' and data_redct == 'PCA':
            skip = True
        if skip == False:
            # debug
            print(runID, crop, algo, forecast_time, doOHE, feature_set, ft_sel, data_redct, addYieldTrend, useSF, aggregationSF)
            with open(os.path.join(config.models_spec_dir, myID + '_' + crop + '_' + algo + '.json'), 'w') as fp:
                json.dump(uset, fp, indent=4)#json.dump(uset, fp)
            runID = runID + 1
