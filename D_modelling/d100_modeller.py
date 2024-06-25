import os
import pandas as pd
import numpy as np
import datetime

from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from sklearn.model_selection import LeaveOneGroupOut #, LeavePGroupsOut, GroupShuffleSplit
#import joblib
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import copy
# from sklearn.preprocessing import KBinsDiscretizer
# from sklearn.base import BaseEstimator, RegressorMixin

from C_model_setting import c1000_utils
from D_modelling import d105_PCA_on_features, d110_benchmark_models, d120_set_hyper, d130_get_hyper, d140_modelStats
# AVOID USE OF MATPLOTLIB GIVING ERRORS IN CONDOR
# from E_viz import e100_eval_figs as viz

# define a data class that it is used to preprocess the data (both in hidcasting and forecasting)
class DataMixin:
    def preprocess(self, config, runType):
        # runType can be:
        # [tuning] tunes models with double LOYO loop (test various configuration)
        # [opeTune] tunes the best model on all available years
        # [opeForecast] run the operational yield

        stats = pd.read_csv(os.path.join(config.models_dir, config.AOI + '_stats.csv'))
        # rescale Production units for better graphic and better models (if production is avail)
        if 'Production' in stats.columns:
            stats['Production'] = stats['Production'].div(config.production_scaler)

        # Raw features for ope forecast are stored in a different dir to avoid overwrite of features used for training
        if runType == 'opeForecast':
            raw_features = pd.read_csv(os.path.join(config.ope_run_dir, config.AOI + '_features4scikit.csv'))
            # drop AU_name, not needed and will be duplicated in merge
            raw_features = raw_features.drop(['AU_name','ASAP1_ID'], axis=1)
            # keep only AU_code that are in stats (some regions may have been dropped because have missing data)
            raw_features = raw_features[raw_features['AU_code'].isin(stats['AU_code'].unique())]
            # the result of merge has year with only features as one record per year while I need one record per crop
            statsDistinct = stats.drop_duplicates(["AU_code", "Crop_ID"])[
                ['Region_ID', 'Crop_ID', 'AU_code', 'AU_name', 'ASAP1_ID', 'Crop_name']]
            raw_features = pd.merge(raw_features, statsDistinct, how='left', left_on=['AU_code'], right_on=['AU_code'])
            # hoin to keep also feature records that have label
            yxData = pd.merge(stats, raw_features, how='outer', left_on=['AU_code', 'Year', 'Crop_ID'],
                              right_on=['AU_code', 'YearOfEOS', 'Crop_ID'])
            yxData[['Region_ID_x', 'AU_name_x', 'ASAP1_ID_x','Crop_name_x']] = yxData[['Region_ID_y', 'AU_name_y', 'ASAP1_ID_y','Crop_name_y']]
            yxData = yxData.drop(['Region_ID_y', 'AU_name_y', 'ASAP1_ID_y','Crop_name_y'], axis=1)
            yxData.rename(columns={'Region_ID_x': 'Region_ID', 'AU_name_x': 'AU_name', 'ASAP1_ID_x': 'ASAP1_ID', 'Crop_name_x': 'Crop_name'}, inplace=True)
            # transfer year of features to year of stats to have it working with trend
            yxData.loc[yxData['Year'].isna(), "Year"] = yxData["YearOfEOS"].astype('int32')
        else: #tuning or opeTune
            raw_features = pd.read_csv(os.path.join(config.models_dir, config.AOI + '_features4scikit.csv'))
            # drop AU_name, not needed and will be duplicated in merge
            raw_features = raw_features.drop(['AU_name','ASAP1_ID'], axis=1)
            # left join to keep only features with labels
            yxData = pd.merge(stats, raw_features, how='left', left_on=['AU_code', 'Year'], right_on=['AU_code', 'YearOfEOS'])

        # retain only the crop to analysed
        yxData = yxData[yxData['Crop_name'] == self.uset['crop']]
        yxData.sort_values(['AU_name', 'Crop_name', 'Year'], ascending=[True, True, True], inplace=True)

        # Add a trend feature (the yield estimate for a year-admin unit)
        if self.uset['algorithm'] == 'Trend' or self.uset['addYieldTrend'] == True:
            yxData = c1000_utils.add_yield_trend_estimate(yxData, self.uset['ny_max_trend'])
        #yxData.to_csv(os.path.join(config.models_dir, 'buttami.csv'), index=False)

        years = yxData['Year']
        AU_codes = yxData['AU_code']
        y = yxData['Yield'].to_numpy()
        #############################################
        # retain features based on feature set/method
        #############################################
        if self.uset['algorithm'] == 'Null_model':
            feature_names = ['None'] # does not have a feature, is simply the mean by AU of targetVar variable
            X = y * 0.0
            X = X.reshape((-1, 1))
        elif self.uset['algorithm'] == 'Trend':
            feature_names = ['None'] # it is the yield estimated by the trend (in X)
            X = yxData['YieldFromTrend'].to_numpy()
            X = X.reshape((-1, 1))
        else: # for any ML model and PeakNDVI
            # retain features up to the time of forecast (included) = lead time
            list2keep = ['(^|\D)M' + str(i) + '($|\D)' for i in range(0, self.uset['forecast_time'] + 1)] + ['YieldFromTrend']
            yxData = yxData.filter(regex='|'.join(list2keep))
            if self.uset['algorithm'] == 'PeakNDVI':
                # the only feature is max NDVI in the period
                feature_names = ['NDpeak']
                X = yxData.filter(like='NDmax').max(axis=1).to_numpy()
                X = X.reshape((-1, 1))
            else:  # ML models and Lasso
                # get the feature group values of the selected feature set
                _features = self.uset['feature_groups']
                # remove not needed features
                if self.uset['addYieldTrend'] == True:
                    list2keep = ['^' + str(i) + 'M\d+$' for i in _features] + ['YieldFromTrend']
                else:
                    list2keep = ['^' + str(i) + 'M\d+$' for i in _features]
                X = yxData.filter(regex='|'.join(list2keep)).to_numpy()
                feature_names = list(yxData.filter(regex='|'.join(list2keep)).columns)
                # now, for Ml models only, scale data if required
                scaler = StandardScaler()  # z-score scaler
                if self.uset['dataScaling'] == 'z_f':       # scale all features
                        X = scaler.fit_transform(X)
                elif self.uset['dataScaling'] == 'z_fl':    # scale all features and label as well
                    X = scaler.fit_transform(X)
                    y = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)  # set in back to (n,)
                elif self.uset['dataScaling']== 'z_fl_au': # scale by AU (both scaling functions checked with xls)
                    y = yxData['Yield'].subtract(yxData.groupby(yxData['AU_code'])['Yield'].transform(np.mean)) \
                        .divide(yxData.groupby(yxData['AU_code'])['Yield'].transform(np.std))
                    y = y.to_numpy()
                    X = yxData[feature_names].subtract(
                        yxData.groupby(yxData['AU_code'])[feature_names].transform(np.mean)) \
                        .divide(yxData.groupby(yxData['AU_code'])[feature_names].transform(np.std))
                    X = X.to_numpy()
                else:
                    print('Data scaling non implemented:  ' + self.uset['dataScaling'])
                    exit()

                # Perform data reduction if requested (PCA)
                # Only on NDVI, RAD, Temp (precipitation, sm are excluded)
                if self.uset['data_reduction'] == 'PCA':
                    X, feature_names = d105_PCA_on_features.getPCA(self, feature_names, X)
                # Perform One-Hot Encoding for AU if requested
            if self.uset['doOHE'] == 'AU_level':
                OHE = pd.get_dummies(AU_codes, columns=['AU_code'], prefix='OHE_AU')
                feature_names = feature_names + OHE.columns.to_list()
                X = np.concatenate((X, OHE), axis=1)
            # End of pre-processing of input data -------------------------------------------
            # Retain selected features for the ope model
        # if runType == 'opeForecast':
        #     print('to be handled in a new way')
        #     exit()
        # now prepare data for hindcasting
        # We use the year as a group (for leave one group out at a time)
        groups = years.to_numpy()
        AU_codes = AU_codes.to_numpy()
        # save the data actually used by the model
        data = pd.DataFrame(
            np.concatenate([AU_codes.reshape((-1, 1)), groups.reshape((-1, 1)), y.reshape((-1, 1)), X], axis=1),
            columns=['AU_code', 'year', 'Yield'] + feature_names)
        myID = self.uset['runID']
        myID = f'{myID:06d}'
        data.to_csv(os.path.join(config.models_out_dir, 'ID_' + str(myID) +
                                 '_crop_' + self.uset['crop'] + '_Yield_' + self.uset['algorithm'] + '_yx_preprocData.csv'), index=False)
        # if runType == 'opeForecast':
        #     # yxData = yxData[yxData'['Year'] == year_out]
        #     ind = groups == year_out
        #     groups = groups[ind]
        #     X = X[ind, :]
        #     y = y[ind]
        #     AU_codes = AU_codes[ind]
        return X, y, groups, feature_names, AU_codes



class YieldModeller(DataMixin, object):
    """
    Yield hindcasting pipeline
    """
    def __init__(self, uset):
        """Instantiates the class with metadata"""
        self.uset = uset
        # print(self.__dict__)
        # print('after YieldModeller init')

    def fit(self, X, y, groups, featureNames, AU_codes, runType):
        # tune and test a model with double loop to avoid information leak

        # X: the feature matrix (n_sample, n_feature)
        # y: the target label (n_sample)
        # groups: an array with a group identification number, the year for example (n_sample),
        #         it is used for leaving one full group at a time
        # featureNames: names of all features passed (n_ferature) (list)
        # au_codes: an array with ancillary data, the AU code for example (n_sample)

        # Df to store predictions
        mRes = pd.DataFrame() #pd.DataFrame(columns=['yLoo_pred', 'yLoo_true', 'AU_code', 'Year'])
        if runType == 'tuning':
            # Outer loop for testing model performances, teh groups are years
            outer_cv = LeaveOneGroupOut()
            gen_outer_cv = outer_cv.split(X, y, groups=groups)
            scoring_metric_on_val = []
            nIterationOuterLoop = 1
            for train_index, test_index in gen_outer_cv:
                #print('Iteration outer loop = ' + str(nIterationOuterLoop))
                X_train, X_test, groups_train, AU_code_train = X[train_index], X[test_index], groups[train_index], AU_codes[train_index]
                y_train, y_test, groups_test, AU_code_test = y[train_index], y[test_index], groups[test_index], AU_codes[test_index]
                # Not implemented:
                # Exclude records (year - AU) with missing nan
                # train set
                nas = np.isnan(y_train)
                y_train = y_train[~nas]
                X_train = X_train[~nas, :]
                groups_train = groups_train[~nas]
                AU_code_train = AU_code_train[~nas]
                # for the test set do nothing, missing values are not an issue (the model does not have to be tuned)

                if self.uset['algorithm'] in ['Null_model', 'Trend', 'PeakNDVI']:
                    outLoopRes = d110_benchmark_models.run_LOYO(self.uset['algorithm'], X_train, X_test, y_train, y_test, AU_code_train, AU_code_test, groups_test)
                    tmp = pd.DataFrame(np.array(outLoopRes).T.tolist(), columns=['yLoo_pred', 'yLoo_true', 'AU_code', 'Year'])
                    mRes = pd.concat([mRes, tmp])
                    #mRes = mRes.append(pd.DataFrame(np.array(outLoopRes).T.tolist(), columns=['yLoo_pred', 'yLoo_true', 'AU_code', 'Year']))
                else:
                    # DEFINE THE INNER LOOP, ITS COMMON TO ALL THE OTHER (ML/LASSO) METHODS
                    inner_cv = LeaveOneGroupOut()
                    if self.uset['feature_selection'] == 'none':
                        n_features = X_train.shape[1]  # used by GPR only
                        search = d120_set_hyper.setHyper(self.uset['algorithm'], self.uset['hyperGrid'], inner_cv,
                                                                   self.uset['nJobsForGridSearchCv'], self.uset['scoringMetric'], n_features=n_features)
                        # Tune hyperparameters
                        search.fit(X_train, y_train, groups=groups_train)

                    elif self.uset['feature_selection'] == 'MRMR':
                        # here feature selection (based on feature_prct_grid) is considered a hyperparam
                        selected_features_names, prct_selected, n_selected, X_test, search = d120_set_hyper.setHyper_ft_sel(
                            X_train, y_train, X_test,
                            self.uset['prct_features2select_grid'], featureNames, groups_train,
                            self.uset['algorithm'], self.uset['hyperGrid'], inner_cv,
                            self.uset['nJobsForGridSearchCv'], self.uset['scoringMetric'])
                    else:
                        print('Feature selection type not implemented in d100')
                        exit()
                    scoring_metric_on_val.append(search.best_score_)
                    # #############################
                    # Now predict the left outs with tuned hypers and store the prediction for the left-out (all years)
                    # if self.uset['algorithm'] == 'XGBoost':
                    #     # final tuning with early stop
                    #     model = search.best_estimator_
                    #     model.early_stopping_rounds = 10
                    #     from sklearn.model_selection import train_test_split
                    #     X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, train_size= 0.95, random_state=0)
                    #     model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)])
                    #     outLoopRes = [model.predict(X_test).tolist(), y_test.tolist(), AU_code_test.tolist(),
                    #                   np.unique(groups_test).tolist() * len(AU_code_test.tolist())]
                    # else:
                    #     outLoopRes = [search.predict(X_test).tolist(), y_test.tolist(), AU_code_test.tolist(),
                    #               np.unique(groups_test).tolist() * len(AU_code_test.tolist())]
                    outLoopRes = [search.predict(X_test).tolist(), y_test.tolist(), AU_code_test.tolist(),
                                  np.unique(groups_test).tolist() * len(AU_code_test.tolist())]
                    tmp = pd.DataFrame(np.array(outLoopRes).T.tolist(), columns=['yLoo_pred', 'yLoo_true', 'AU_code', 'Year'])
                    mRes = pd.concat([mRes, tmp])
                    # check and store if best params are pegged to boundaries
                    if nIterationOuterLoop == 1:
                        # Define variable to keep track on pegging to grid boundaries
                        nPeggedLeft = dict((el, 0) for el in self.uset['hyperGrid'].keys())
                        nPeggedRight = copy.deepcopy(nPeggedLeft)
                    keys = list(search.best_params_.keys())
                    for k in keys:
                        if not isinstance(self.uset['hyperGrid'][k], int):
                            if search.best_params_[k] == self.uset['hyperGrid'][k][0]:
                                nPeggedLeft[k] = nPeggedLeft[k] + 1
                            if search.best_params_[k] == self.uset['hyperGrid'][k][-1]:
                                nPeggedRight[k] = nPeggedRight[k] + 1
                        else:
                            if search.best_params_[k] == self.uset['hyperGrid'][k].low:
                                nPeggedLeft[k] = nPeggedLeft[k] + 1
                            if search.best_params_[k] == self.uset['hyperGrid'][k].high:
                                nPeggedRight[k] = nPeggedRight[k] + 1

                nIterationOuterLoop += 1
                # End of outer script

        #print('d100 end of outer loop')
        # Here the outer loop (activate in tuining) is concluded and I have all results

        # Fitting stats and summary of nPegged
        # For the fitting I apply the model to all data available
        # Fitting is also done to collect hyper for the operational model to be use for real forecasts
        # Set default results (used by benchmark, and overwritten in case peak or ML are used
        search = 'benchmark with no serach in d100_modeller'
        hyperParamsGrid = np.nan
        hyperParams = np.nan
        Fit_R2 = np.nan
        coefFit = np.nan
        selected_features_names = featureNames #set it to all, so if the is no feature selection they are unchanged
        prct_selected, n_selected = '', ''
        avg_scoring_metric_on_val = np.nan
        prctPegged = {'left': '', 'right': ''}

        if (self.uset['algorithm']  == 'PeakNDVI'):
            y_true, y_pred, search = d110_benchmark_models.run_fit(self.uset['algorithm'], X, y, AU_codes)
            # search is the model to be used in prediction
            Fit_R2 = d140_modelStats.r2_nan(np.array(y_true), np.array(y_pred))
        elif not(self.uset['algorithm'] in ['Null_model', 'Trend']): #it is a ML model
        #else: #it is a ML model
            # regenerate an outer loop for setting hyperparameters
            outer_cv = LeaveOneGroupOut()
            nas = np.isnan(y)
            y = y[~nas]
            X = X[~nas, :]
            groups = groups[~nas]
            if self.uset['feature_selection'] == 'none':
                n_features = X.shape[1]  # used by GPR only
                search = d120_set_hyper.setHyper(self.uset['algorithm'], self.uset['hyperGrid'], outer_cv,
                                                           self.uset['nJobsForGridSearchCv'],
                                                           self.uset['scoringMetric'], n_features=n_features)
                # Tune hyperparameters
                search.fit(X, y, groups=groups)
                Fit_R2 = metrics.r2_score(y, search.predict(X))
                # selected_features_names = featureNames
            elif self.uset['feature_selection'] == 'MRMR':
                # here feature selection (based on feature_prct_grid) is considered a hyperparam
                selected_features_names, prct_selected, n_selected, X_2use, search = d120_set_hyper.setHyper_ft_sel(
                    X, y, X,
                    self.uset['prct_features2select_grid'], featureNames,
                    groups,
                    self.uset['algorithm'], self.uset['hyperGrid'], outer_cv,
                    self.uset['nJobsForGridSearchCv'], self.uset['scoringMetric'])
                Fit_R2 = metrics.r2_score(y, search.predict(X_2use))  # It is X test because it is X with only the selected features
            # now get hyperparams (each model has its own)
            if runType == 'tuning':
                prctPegged = {'left': '', 'right': ''}
                nIterationOuterLoop = nIterationOuterLoop - 1  # remove last increment at the the end of loop (the increment is at the end of the loop)
                for (key, value) in nPeggedLeft.items():
                    if value > 0:
                        prctPegged['left'] = prctPegged['left'] + key + ':' + str(round(value / nIterationOuterLoop * 100)) + ';'
                for (key, value) in nPeggedRight.items():
                    if value > 0:
                        prctPegged['right'] = prctPegged['right'] + key + ':' + str(round(value / nIterationOuterLoop * 100)) + ';'
                avg_scoring_metric_on_val = np.mean(
                    [-x for x in scoring_metric_on_val])  # rmse is negative, turn it positive
            else:
                # pegging check was skipped, as the model was run in fit only. give a nana value
                prctPegged = np.nan
                avg_scoring_metric_on_val = np.nan
            hyperParams, hyperParamsGrid, coefFit = d130_get_hyper.get(self.uset['algorithm'], search, self.uset['hyperGrid'],
                                       selected_features_names)

        #print(self.__dict__)
        #print('d100 YieldModeller ended')
        return hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, prctPegged, selected_features_names, prct_selected, n_selected, avg_scoring_metric_on_val, search

    def validate(self, hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, prctPegged, runTimeH, featureNames,
                 selected_features_names, prct_selected, n_selected, avg_scoring_metric_on_val, config, save_file=True,
                 save_figs=False):
        error_spatial = d140_modelStats.allStats_spatial(mRes)
        error_overall = d140_modelStats.allStats_overall(mRes)
        # mean of temporal R2 of each AU
        meanAUR2 = d140_modelStats.meanAUR2(mRes)  # equivalent to R2 within
        # National level stats
        stats = pd.read_csv(os.path.join(config.models_dir, config.AOI + '_stats.csv'))
        # national yield using subnat yield weighted by area
        tmp = stats[stats['Crop_name'] == self.uset['crop']][['Region_ID', 'Area']]
        # get avg area
        avg_area = tmp.groupby('Region_ID').mean()
        def weighed_average(grp):
            return grp._get_numeric_data().multiply(grp['Area'], axis=0).sum() / grp['Area'].sum()

        mRes = pd.merge(mRes, avg_area, how='left', left_on=['AU_code'], right_on=['Region_ID'])
        mCountryRes = mRes.groupby('Year')[['yLoo_pred', 'yLoo_true', 'Area']].apply(weighed_average).drop(
            ['Area'], axis=1)
        error_Country_level = d140_modelStats.allStats_country(mCountryRes)

        # store results in dataframe
        outdict = {'runID': str(self.uset['runID']),
                   'dataScaling': self.uset['dataScaling'], # if self.model_name not in cst.benchmarks else '',
                   'DoOHEnc': self.uset['doOHE'], # if self.model_name not in cst.benchmarks else '',
                   #'AddTargetMeanToFeature': cst.AddTargetMeanToFeature if self.model_name not in cst.benchmarks else '',
                   'AddYieldTrend': str(self.uset['addYieldTrend']), # if self.model_name not in cst.benchmarks else '', #cst.DoScaleOHEnc if self.model_name not in cst.benchmarks else '',
                   'scoringMetric': self.uset['scoringMetric'],# if self.model_name not in cst.benchmarks else '',
                   # 'n_out_inner_loop': cst.n_out_inner_loop,
                   'nJobsForGridSearchCv': self.uset['nJobsForGridSearchCv'],
                   'Time': datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                   #'Time_sampling': self.time_sampling,
                   'forecast_time': self.uset['forecast_time'],
                   'N_features': str(len(featureNames)), # if self.model_name not in cst.benchmarks else '',
                   #'N_OHE': str(self.nOHE),
                   'Data_reduction': str(self.uset['data_reduction']),
                   'Feature_set': str(self.uset['feature_set']),
                   'Features': [featureNames],
                   'Ft_selection' : self.uset['feature_selection'],
                   'N_selected_fit': n_selected,
                   'Prct_selected_fit': prct_selected,
                   'Selected_features_names_fit': [selected_features_names],
                   #'targetVar': self.yvar,
                   'Crop': self.uset['crop'],
                   'Estimator': self.uset['algorithm'],
                   #'Optimisation': self.optimisation,
                   'RMSE_val': avg_scoring_metric_on_val,
                   'R2_f': Fit_R2,
                   'avg_R2_p_overall': error_overall['Pred_R2'],
                   'avg_R2_p_spatial': error_spatial['Pred_R2'],
                   'avg_R2_p_temporal(alias R2_WITHINp)': str(meanAUR2),
                   'MAE_p': error_overall['Pred_MAE'],
                   'rMAE_p': error_overall['rel_Pred_MAE'],
                   'ME_p': error_overall['Pred_ME'],
                   'RMSE_p': error_overall['Pred_RMSE'],
                   'rRMSE_p': error_overall['rel_Pred_RMSE'],
                   'HyperParGrid': hyperParamsGrid,
                   'HyperPar': hyperParams,
                   'Country_R2_p': error_Country_level['Pred_R2'],
                   'Country_MAE_p': error_Country_level['Pred_MAE'],
                   'Country_ME_p': error_Country_level['Pred_ME'],
                   'Country_RMSE_p': error_Country_level['Pred_RMSE'],
                   'Country_rRMSE_p': error_Country_level['rel_Pred_RMSE'],
                   'Country_FQ_RMSE_p': error_Country_level['Pred_RMSE_FQ'],
                   'Country_FQ_rRMSE_p': error_Country_level['Pred_rRMSE_FQ'],
                   'Mod_coeff': coefFit,
                   '%TimesPegged_left': prctPegged['left'],
                   '%TimesPegged_right': prctPegged['right'],
                   'run time (h)': runTimeH
        }
        res_df = pd.DataFrame.from_dict(outdict)
        if save_file:
            myID = self.uset['runID']
            myID = f'{myID:06d}'
            res_df.to_csv(os.path.join(config.models_out_dir, 'ID_' + str(myID) +
                                '_crop_' + self.uset['crop'] + '_Yield_' + self.uset['algorithm'] +
                                '_output.csv'), index=False)
        # AVOID USE OF MATPLOTLIB GIVING ERRORS IN CONDOR
        # if save_figs:
        #     accuracy_over_time = os.path.join(config.models_out_dir, 'ID_' + str(myID) +
        #                         '_crop_' + self.uset['crop'] + '_Yield_' + self.uset['algorithm'] + '_accuracy_over_time.png')
        #     viz.accuracy_over_time(mRes, mCountryRes, filename=accuracy_over_time)
        #
        #     scatter_plot_au = os.path.join(config.models_out_dir, 'ID_' + str(myID) +
        #                         '_crop_' + self.uset['crop'] + '_Yield_' + self.uset['algorithm'] + '_scatter_by_AU.png')
        #     viz.scatter_plot_accuracy(mRes, error_spatial['Pred_R2'], "AU_code", filename=scatter_plot_au)
        #     scatter_plot_year = os.path.join(config.models_out_dir, 'ID_' + str(myID) +
        #                         '_crop_' + self.uset['crop'] + '_Yield_' + self.uset['algorithm'] + '_scatter_by_year.png')
        #     viz.scatter_plot_accuracy(mRes, error_spatial['Pred_R2'], "Year", filename=scatter_plot_year)
        # return