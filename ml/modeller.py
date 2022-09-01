import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import os
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.decomposition import PCA
from scipy import stats
import re

import src.constants as cst
import preprocess.b1000_preprocess_utilities as b1000_preprocess_utilities
import b05_Init
#import b110_modelApp
import b110_modelApp_with_feature_selection
import Model_error_stats
import viz.evaluation as viz


def add_yield_trend_estimate(yxDatac, yvar):
    """
    Add a feature YFromTrend (the y value estimated by the y time trend of a given admin unit).
    The type of trend computation is different depending on how long is the time series of stats data we have.
    if we have at least ny years of stat before timeRange[0] we use ny before year YYYY to estimate yield of YYYY
    else (less than ny years) we use Franz's idea:
    to avoid using info before AND after (not correct because not possible in NRT), we assume that there is no difference between forward and backward y trend estimates,
    meaning that y of year YYYY can be estimated from the time series YYYY-n : YYYY-1 or from the one YYYY+1 : YYYY+n
    So, we locate the year YYYY in the time series. It will have nb year before and na year after.
    We use the years after (if na>nb) or before (if nb>na) to estimate the trend and compute the yield of YYYY.
    Note: computing the y trend estimate beforehand (not in the double loop) has the drawback that when holding out year YYYY (e.g. 2005),
    the model will be tuned with a trend feature computed using YYYY for the years after YYYY (e.g. 2006, 2007, ..).
    This could be avoided computing the trend in the loop and thus excluding YYYY of the outer loop and YYYY2 of inner loop.
    However: 1) with scikit CV for hyperpar optimisation we do not have access to inner loop data, and 2) although not using YYYY,
    we would often use data before and after YYYY (e.g. when YYYY is 2005 and we are estimating 2009).
    Therefore, we opt for some form of info leakage that is due to precomputing Y trend of YYYY. Nevertheless, as in operation,
    data after YYYY are never used to estimate its y value from the trend.
    """
    # number of years of y data before the first year with features AND number of years for trend computation
    # must be sourced from constant
    ny = cst.ny_max_trend

    def trend1(row, ny, minYearFeats):
        """
        This function computes the trend for a given row using ny year before
        """
        if row['Year'] < minYearFeats:
            # do nothing for years without features
            return np.nan
        else:
            # compute the trend
            years2use = list(range(row['Year']-ny, row['Year']))
            y = row[years2use].values.astype(float)
            x = np.array(years2use).astype(float)
            ind = ~ np.logical_or(np.isnan(y), np.isnan(x))
            res = stats.theilslopes(y[ind], x[ind])
            return res[1] + res[0] * row['Year']

    def trend2(row, ny, minYearFeats, yearList):
        """
        This function computes the trend for a given row using the longest time series (left or right to the year to estimate)
        """
        yearList = np.array(yearList)
        if row['Year'] < minYearFeats:
            # do nothing for years without features
            return np.nan
        else:
            # find the largest arm of the time series (before or after)
            leftYears = yearList[yearList < row['Year']]
            rightYears = yearList[yearList > row['Year']]
            if len(leftYears) >= len(rightYears):
                #use the left arm
                years2use = leftYears
                if len(years2use) > ny:
                    #keep only ny, the last ones
                    years2use = years2use[-ny:]
            else:
                #use the right arm
                years2use = rightYears
                if len(years2use) > ny:
                    #keep only ny, the first ones
                    years2use = years2use[0:ny]
            y = row[years2use].values.astype(float)
            x = np.array(years2use).astype(float)
            ind = ~ np.logical_or(np.isnan(y), np.isnan(x))
            res = stats.theilslopes(y[ind], x[ind])
            return res[1] + res[0] * row['Year']

    yxDatac['YieldFromTrend'] = np.nan
    # treat each crop / region separately as the stat availability may be different
    for c in yxDatac['Crop_ID'].unique():
        for r in yxDatac['AU_code'].unique():
            df = yxDatac[(yxDatac['Crop_ID'] == c) & (yxDatac['AU_code'] == r)]
            minYearStats = df.dropna(subset=[yvar])['Year'].min()
            minYearFeats = df.dropna(subset=df.columns[~df.columns.isin(['YieldFromTrend'])].values)['Year'].min()
            df[df['Year'].tolist()] = df[yvar].values
            # use different trend if we have more than ny before the first feature data point
            if minYearFeats - minYearStats > ny:
                # trend estimated with ny years before
                df['YieldFromTrend'] = df.apply(trend1, args=(ny, minYearFeats), axis=1)
            else:
                # trend estimated using larger time series (lef or right)
                df['YieldFromTrend'] = df.apply(trend2, args=(ny, minYearFeats, df['Year'].tolist()), axis=1)
            # add the trend to yxDatac
            yxDatac.loc[(yxDatac['Crop_ID'] == c) & (yxDatac['AU_code'] == r), 'YieldFromTrend'] = df['YieldFromTrend']
    return yxDatac

class YieldHindcaster(object):
    """
    Yield hindcasting pipeline
    """

    def __init__(self, run_id, aoi, crop, algo, yvar, feature_set, feature_selection, data_reduction,
                 prct_features2select_grid, n_features2select_grid, doOHE, forecast_time, yieldTrend, time_sampling):
        """Instantiates the class with metadata"""
        self.useTrend = yieldTrend
        self.id = run_id
        self.aoi = aoi
        self.model_name = algo
        self.crop = crop
        self.crop_name = ''
        self.yvar = yvar
        self.leadtime = forecast_time
        self.doOHE = doOHE
        self.nOHE = 0
        self.time_sampling = time_sampling
        self.optimisation  = cst.hyperparopt

        # Derive other parameters from src.constants
        if algo in cst.hyperGrid.keys():
            self.hyperparams = cst.hyperGrid[algo]
        else:
            self.hyperparams = None

        self.feature_prct_grid = cst.feature_prct_grid
        self.prct_features2select_grid = prct_features2select_grid
        self.n_features2select_grid = n_features2select_grid
        self.metric = cst.scoringMetric
        self.feature_group = cst.feature_groups[feature_set]
        self.feature_selection = feature_selection
        self.data_reduction = data_reduction
        self.PCAprctVar2keep = cst.PCAprctVar2keep
        self.feature_fn = None
        self.model_fn = None
        stamp_id = run_id.split('_')[0]
        self.output_dir = os.path.join(cst.odir, self.aoi, 'Model', stamp_id)
        self.output_dir_mres = os.path.join(self.output_dir, 'mres')
        self.output_dir_yx_preprocData = os.path.join(self.output_dir, 'yx_preprocData')
        self.output_dir_output = os.path.join(self.output_dir, 'output')
        self.figure_dir = os.path.join(self.output_dir, 'figures')
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir_mres).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir_yx_preprocData).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir_output).mkdir(parents=True, exist_ok=True)
        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)

    def preprocess(self, save_to_csv=True):
        project = b05_Init.init(self.aoi)
        prct2retain = project['prct2retain']
        statsX = pd.read_pickle(os.path.join(cst.odir, self.aoi, f'{self.aoi}_stats{prct2retain}.pkl'))  # stats for the 90% main prodducers
        #statsX = pd.read_pickle(os.path.join(cst.odir, self.aoi, f'{self.aoi}_stats90.pkl'))
        stats = pd.read_pickle(os.path.join(cst.odir, self.aoi, f'{self.aoi}_stats.pkl'))  # all targetCountry stats
        # drop years before period of interest
        # Mic 20210518 for trend computation remove stats = stats.drop(stats[stats['Year'] < cst.timeRange[0]].index), cst.timeRange should not be used
        # drop unnecessary column
        stats = stats.drop(cst.drop_cols_stats, axis=1)
        # rescale Production units for better graphic and better models
        stats['Production'] = stats['Production'].div(cst.production_scaler)

        # open predictors
        raw_features = pd.read_pickle(os.path.join(cst.odir, self.aoi, f'{self.aoi}_pheno_features4scikit.pkl'))

        # merge stats and features, so that at ech stat entry I have the full set of features
        yxData = pd.merge(stats, raw_features, how='left', left_on=['AU_code', 'Year'],
                          right_on=['AU_code', 'YearOfEOS'])

        yxDatac = b1000_preprocess_utilities.retain_X(yxData, statsX, self.crop)
        self.crop_name = yxDatac['Crop_name'].iloc[0]

        # Add a trend feature (the yield estimate for a year-admin unit)
        if self.model_name == 'Trend' or self.useTrend == True:
            yxDatac = add_yield_trend_estimate(yxDatac, self.yvar)
        # Remove years having only statistics and not features (use to compute the trend estimate of the yield
        yxDatac = yxDatac.drop(yxDatac[yxDatac['Year'] < project['timeRange'][0]].index)

        # Get labels (i.e. y values) and ancillary info as separate df
        labels = yxDatac[['Yield', 'Production']]
        years = yxDatac['Year']
        AU_codes = yxDatac['AU_code']

        #############################################
        # retain features based on feature set/method
        #############################################
        if self.model_name == 'Null_model':
            feature_names = ['None']
            # does not have a feature, is simply the mean by AU of targetVar variable
            y = labels[self.yvar].to_numpy()
            X = y * 0.0
            X = X.reshape((-1, 1))
        elif self.model_name == 'Trend':
            feature_names = ['None']
            # is the yield estimated by the trend (in X)
            y = labels[self.yvar].to_numpy()
            X = yxDatac['YieldFromTrend'].to_numpy()
            X = X.reshape((-1, 1))
        else: # for any ML model and PeakNDVI
            # retain features up to the time of forecast (included) = lead time
            list2keep = ['(^|\D)' + self.time_sampling + str(i) + '($|\D)' for i in range(0, self.leadtime + 1)] + ['YieldFromTrend']
            yxDatac = yxDatac.filter(regex='|'.join(list2keep))
            if self.model_name == 'PeakNDVI':
                # the only feature is max NDVI in the period
                feature_names = ['NDpeak']
                X = yxDatac.filter(like='NDmax').max(axis=1).to_numpy()
                X = X.reshape((-1, 1))
                y = labels[self.yvar].to_numpy()
            else:  # ML models and Lasso
                y = labels[self.yvar].to_numpy()
                # remove not needed features
                _features = [s + self.time_sampling for s in self.feature_group]
                if self.useTrend == True:
                    list2keep = [str(i) + '(?!\D+)' for i in _features] + ['YieldFromTrend']
                else:
                    list2keep = [str(i) + '(?!\D+)' for i in _features]

                X = yxDatac.filter(regex='|'.join(list2keep)).to_numpy()
                feature_names = list(yxDatac.filter(regex='|'.join(list2keep)).columns)

            # Preprocessing of input variables using z-score
            if self.model_name not in cst.benchmarks:
                scaler = StandardScaler()  # z-score scaler
                if cst.dataScaling == 'z_f':
                    # scale all features
                    X = scaler.fit_transform(X)
                elif cst.dataScaling == 'z_fl':
                    # scale all features and label as well
                    X = scaler.fit_transform(X)
                    y = scaler.fit_transform(y.reshape(-1, 1))
                    y = y.reshape(-1)  # set in back to (n,)
                elif cst.dataScaling == 'z_fl_au':
                    # scale by AU (both scaling functions checked with xls)
                    y = yxDatac[self.yvar].subtract(yxDatac.groupby(yxDatac['AU_code'])[self.yvar].transform(np.mean)) \
                        .divide(yxDatac.groupby(yxDatac['AU_code'])[self.yvar].transform(np.std))
                    y = y.to_numpy()
                    X = yxDatac[feature_names].subtract(yxDatac.groupby(yxDatac['AU_code'])[feature_names].transform(np.mean)) \
                        .divide(yxDatac.groupby(yxDatac['AU_code'])[feature_names].transform(np.std))
                    X = X.to_numpy()
                else:
                    print(f'data scaling non implemented:  {cst.dataScaling}')
                    exit()
            # Perform data reduction if requested
            # Only on NDVI, RAD, Temp (precipitation is excluded)
            if self.data_reduction == 'PCA':
                # first: - no request of PCA on one single month should arrive here (skipped in b100)
                #        - z scaling is done already above
                # second: operate PCA on all var of group except RainSum [ 0-9]
                # get list of feature type in feauture group and exclude RainSum
                feature2PCA = [s for s in self.feature_group if s != 'RainSum'] #[f(x) for x in sequence if condition]
                for var2PCA in feature2PCA:
                    # print(var2PCA)
                    # print('shape in', X.shape)
                    # print('varin', feature_names)
                    idx2PCAlist = [i for i, item in enumerate(feature_names) if re.search(var2PCA+self.time_sampling + '[0-9]+', item)]
                    var2PCAlist = list(np.array(feature_names)[idx2PCAlist])
                    v = X[:,idx2PCAlist]
                    #perform PCA and keep the required variance fraction
                    n_comp = len(var2PCAlist)-1 #at least one dimension less
                    pca = PCA(n_components=n_comp, svd_solver='full')
                    Xpca = pca.fit_transform(v)
                    # retain components up to PCAprctVar2keep, if this is never reached take all
                    if np.cumsum(pca.explained_variance_ratio_)[-1] <= cst.PCAprctVar2keep/100:
                        indexComp2retain = n_comp-1
                    else:
                        indexComp2retain = np.argwhere(np.cumsum(pca.explained_variance_ratio_)>cst.PCAprctVar2keep/100)[0][0]
                        # print(pca.explained_variance_ratio_)
                    # print(indexComp2retain)
                    Xpca = Xpca[:,0:indexComp2retain+1]
                    # now replace original columns with PCAs
                    new_features_names = [var2PCA+'_PCA'+str(s) for s in range(1,indexComp2retain+1+1)]
                    X =np.delete(X, idx2PCAlist, 1)
                    feature_names = list(np.delete(np.array(feature_names),idx2PCAlist))
                    feature_names = feature_names + new_features_names
                    X = np.concatenate((X, Xpca), axis=1)
                    # print('shape out', X.shape)
                    # print('varout', feature_names)
                    # print()
                # now adjust n feature to be secelted in case feature selction is required
                if self.feature_selection != 'none':
                    if len(feature_names) == 1:
                        #PCA has reduced to 1 feature, no feature selection possible
                        self.feature_selection = 'none'
                    else:
                        # print(self.prct_features2select_grid)
                        # print(self.n_features2select_grid)
                        n_features = len(feature_names)
                        prct_grid = cst.feature_prct_grid
                        n_features2select_grid = n_features * np.array(prct_grid) / 100
                        n_features2select_grid = np.round_(n_features2select_grid, decimals=0, out=None)
                        # keep those with at least 1 feature
                        idx2retain = [idx for idx, value in enumerate(n_features2select_grid) if value >= 1]
                        prct_features2select_grid = np.array(prct_grid)[idx2retain]
                        n_features2select_grid = n_features2select_grid[idx2retain]
                        # drop possible duplicate in n, and if the same number of ft referes to multiple %, take the largest (this explain the np.flip)
                        n_features2select_grid, idx2retain = np.unique(np.flip(np.array(n_features2select_grid)),
                                                                       return_index=True)
                        prct_features2select_grid = np.flip(prct_features2select_grid)[idx2retain]
                        self.prct_features2select_grid = prct_features2select_grid
                        self.n_features2select_grid = n_features2select_grid
            # Perform One-Hot Encoding for AU if requested
            if self.doOHE == 'AU_level':
                OHE = pd.get_dummies(AU_codes, columns=['AU_code'], prefix='OHE_AU')
                feature_names = feature_names + OHE.columns.to_list()
                self.nOHE = len(OHE.columns.to_list())
                OHE = OHE.to_numpy()
                if cst.DoScaleOHEnc:
                    OHE = scaler.fit_transform(OHE)
                X = np.concatenate((X, OHE), axis=1)
            elif self.doOHE == 'Cluster_level':
                # replace column "AU_code" by the cluster id for OHE
                clustering = pd.read_csv(os.path.join(cst.odir, self.aoi,
                                                      'AU_clusters', f'{self.crop_name}_clusters.csv'))
                AU_codes = pd.merge(AU_codes, clustering, left_on='AU_code', right_on='AU_code')
                OHE = pd.get_dummies(AU_codes['Cluster_ID'], columns=['Cluster_ID'], prefix='OHE_Cluster')
                feature_names = feature_names + OHE.columns.to_list()
                self.nOHE = len(OHE.columns.to_list())
                OHE = OHE.to_numpy()
                if cst.DoScaleOHEnc:
                    OHE = scaler.fit_transform(OHE)
                X = np.concatenate((X, OHE), axis=1)
                #set back AU_codes to only AU_codes
                AU_codes = AU_codes['AU_code']
        # End of pre-processing of input data -------------------------------------------

        # We use the year as a group (for leave one group out at a time)
        groups = years.to_numpy()
        AU_codes = AU_codes.to_numpy()
        # save the data actually used by the model
        data = pd.DataFrame(
            np.concatenate([AU_codes.reshape((-1, 1)), groups.reshape((-1, 1)), y.reshape((-1, 1)), X], axis=1),
            columns=['AU_code', 'year', self.yvar] + feature_names)
        if save_to_csv:
            data.to_csv(os.path.join(self.output_dir_yx_preprocData, f'ID_{self.id}_crop_{self.crop}_{self.yvar}_{self.model_name}_yx_preprocData.csv'))
        return X, y, groups, feature_names, AU_codes

    def fit(self, X, y, groups, feature_names, AU_codes, save_output):
        hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, nPegged, selected_features_names, prct_selected, n_selected, avg_scoring_metric_on_val = \
            b110_modelApp_with_feature_selection.modelApp(X, y, groups, feature_names, AU_codes,  #b110_modelApp.modelApp(X, y, groups, feature_names, AU_codes,
                                   self.model_name,
                                   cst.AddTargetMeanToFeature,
                                   self.metric,
                                   self.optimisation, self.hyperparams,
                                   cst.n_out_inner_loop,
                                   cst.nJobsForGridSearchCv,
                                   feature_selection=self.feature_selection,
                                   prct_features2select_grid=self.prct_features2select_grid,
                                   n_fetures2select_grid=self.n_features2select_grid)
        if save_output:
            strFn = os.path.join(self.output_dir_mres, f'ID_{self.id}_crop_{self.crop}_{self.yvar}_{self.model_name}_mRes.csv')
            mRes = mRes.astype(np.float32)
            mRes.to_csv(strFn.replace(" ", ""))
        return hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, nPegged, selected_features_names,  prct_selected, n_selected, avg_scoring_metric_on_val



    def validate(self, hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, nPegged, runTimeH, featureNames,
                 selected_features_names,  prct_selected, n_selected, avg_scoring_metric_on_val, save_file=True, save_figs=False):

        def weighed_average(grp):
            return grp._get_numeric_data().multiply(grp['Production'], axis=0).sum() / grp['Production'].sum()

        #error_AU_level = Model_error_stats.allStats(mRes, Compute_Pred_MrAE=True)
        error_AU_level = Model_error_stats.allStats(mRes)
        meanAUR2 = Model_error_stats.meanAUR2(mRes)  # equivalent to R2 within

        # National level stats
        if self.yvar == 'Production':
            mCountryRes = mRes.groupby('Year')[['yLoo_pred', 'yLoo_true']].mean()
        if self.yvar == 'Yield':
            # here I  weight the yield based on mean production
            # get AU mean production
            project = b05_Init.init(self.aoi)
            prct2retain = project['prct2retain']
            statsX = pd.read_pickle(os.path.join(cst.odir, self.aoi, f'{self.aoi}_stats{prct2retain}.pkl'))
            #statsX = pd.read_pickle(os.path.join(cst.odir, self.aoi, f'{self.aoi}_stats90.pkl'))
            tmp = statsX[statsX['Crop_ID'] == self.crop][[('Region_ID', ''),
                                                            ('Production', 'mean')]].droplevel(1, axis=1)
            mRes = pd.merge(mRes, tmp, how='left', left_on=['AU_code'], right_on=['Region_ID'])
            mCountryRes = mRes.groupby('Year')[['yLoo_pred', 'yLoo_true', 'Production']].apply(weighed_average).drop(
                ['Production'], axis=1)
        error_Country_level = Model_error_stats.allStats_country(mCountryRes)

        # store results in dataframe
        outdict = {'runID': str(self.id),
                   'dataScaling': cst.dataScaling if self.model_name not in cst.benchmarks else '',
                   'DoOHEnc': self.doOHE if self.model_name not in cst.benchmarks else '',
                   'AddTargetMeanToFeature': cst.AddTargetMeanToFeature if self.model_name not in cst.benchmarks else '',
                   'AddYieldTrend': str(self.useTrend) if self.model_name not in cst.benchmarks else '', #cst.DoScaleOHEnc if self.model_name not in cst.benchmarks else '',
                   'scoringMetric': cst.scoringMetric if self.model_name not in cst.benchmarks else '',
                   'n_out_inner_loop': cst.n_out_inner_loop,
                   'nJobsForGridSearchCv': cst.nJobsForGridSearchCv,
                   'Time':datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                   'Time_sampling': self.time_sampling,
                   'forecast_time': self.leadtime,
                   'N_features': str(len(featureNames)) if self.model_name not in cst.benchmarks else '',
                   'N_OHE': str(self.nOHE),
                   'Data_reduction': str(self.data_reduction),
                   'Features': [featureNames],
                   'Ft_selection' : self.feature_selection,
                   'N_selected_fit': n_selected,
                   'Prct_selected_fit': prct_selected,
                   'Selected_features_names_fit': [selected_features_names],
                   'targetVar': self.yvar,
                   'Crop': self.crop_name,
                   'Estimator': self.model_name,
                   'Optimisation': self.optimisation,
                   'R2_f': Fit_R2,
                   'RMSE_val': avg_scoring_metric_on_val,
                   'R2_p': error_AU_level['Pred_R2'],
                   'MAE_p': error_AU_level['Pred_MAE'],
                   'rMAE_p': error_AU_level['rel_Pred_MAE'],
                   'ME_p': error_AU_level['Pred_ME'],
                   'RMSE_p': error_AU_level['Pred_RMSE'],
                   'rRMSE_p': error_AU_level['rel_Pred_RMSE'],
                   'HyperParGrid': hyperParamsGrid,
                   'HyperPar': hyperParams,
                   'avg_AU_R2_p(alias R2_WITHINp)': str(meanAUR2),
                   'Country_R2_p': error_Country_level['Pred_R2'],
                   'Country_MAE_p': error_Country_level['Pred_MAE'],
                   'Country_ME_p': error_Country_level['Pred_ME'],
                   'Country_RMSE_p': error_Country_level['Pred_RMSE'],
                   'Country_rRMSE_p': error_Country_level['rel_Pred_RMSE'],
                   'Country_FQ_RMSE_p': error_Country_level['Pred_RMSE_FQ'],
                   'Country_FQ_rRMSE_p': error_Country_level['Pred_rRMSE_FQ'],
                   'Mod_coeff': coefFit,
                   '%TimesPegged_left': nPegged['left'],
                   '%TimesPegged_right': nPegged['right'],
                   'run time (h)': runTimeH
        }
        res_df = pd.DataFrame.from_dict(outdict)
        if save_file:
            self.model_fn = os.path.join(self.output_dir_output, f'ID_{self.id}_crop_{self.crop}_{self.yvar}_{self.model_name}_output.csv')
            res_df.to_csv(self.model_fn, index=False)

        # plot
        if save_figs:
            accuracy_over_time = os.path.join(self.figure_dir,
                                              f'ID_{self.id}_crop_{self.crop}_{self.yvar}_{self.model_name}.png')
            viz.accuracy_over_time(mRes, mCountryRes, filename=accuracy_over_time)

            scatter_plot_au = os.path.join(self.figure_dir,
                                           f'ID_{self.id}_crop[_{self.crop}_{self.yvar}_{self.model_name}_scatter_by_AU.png')
            viz.scatter_plot_accuracy(mRes, error_AU_level['Pred_R2'], "AU_code", filename=scatter_plot_au)

            scatter_plot_year = os.path.join(self.figure_dir,
                                             f'ID_{self.id}_crop_{self.crop}_{self.yvar}_{self.model_name}_scatter_by_year.png')
            viz.scatter_plot_accuracy(mRes, error_AU_level['Pred_R2'], "Year", filename=scatter_plot_year)

        return res_df

    def __str__(self):
        return f'------------------------\n' \
               f'The details of the pipeline are: \n' \
               f'Model run ID           : {self.id}\n' \
               f'Area of interest       : {self.aoi}\n' \
               f'Crop type              : {self.crop}\n' \
               f'Algorithm              : {self.model_name}\n' \
               f'Dependent variable     : {self.yvar}\n' \
               f'Time of forecast       : {self.leadtime}\n' \
               f'Time for sampling      : {self.time_sampling} \n' \
               f'Metric to optimise     : {self.metric} \n' \
               f'Feature set            : {self.feature_group}\n' \
               f'One hot Encoding       : {self.doOHE}\n' \
               f'Use trend              : {self.useTrend } \n' \
               f'Ft selection           : {self.feature_selection} \n' \
               f'Data reduction         : {self.data_reduction} \n' \
               f'Path to model inputs   : {self.feature_fn}\n' \
               f'Path to model          : {self.model_fn}\n' \
               f'------------------------'


class YieldForecaster(object):
    """
    Yield forecasting pipeline
    """

    def __init__(self, run_id, aoi, crop, algo, yvar, doOHE, selected_features, forecast_time, time_sampling):
        """Instantiates the class with metadata"""
        self.id = run_id
        #self.firstYear = firstYear
        self.aoi = aoi
        self.model_name = algo
        self.crop = crop
        self.crop_name = ''
        self.yvar = yvar
        self.leadtime = forecast_time
        self.doOHE = doOHE
        self.OHE = None
        self.time_sampling = time_sampling
        self.optimisation = cst.hyperparopt

        # Derive other parameters from src.constants
        if algo in cst.hyperGrid.keys():
            self.hyperparams = cst.hyperGrid[algo]
        else:
            self.hyperparams = None

        self.AUs = None
        self.selected_features = selected_features
        self.scaler = None
        self.metric = cst.scoringMetric
        self.scaler = StandardScaler()  # z-score scaler
        self.search = None

        self.feature_fn = None
        self.model_fn   = None
        stamp_id = run_id.split('_')[0]
        self.output_dir = os.path.join(cst.odir, self.aoi, 'Model', stamp_id)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.figure_dir = os.path.join(cst.odir, self.aoi, 'figures')
        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)

    def preprocess(self):
        project = b05_Init.init(self.aoi)
        prct2retain = project['prct2retain']
        statsX = pd.read_pickle(os.path.join(cst.odir, self.aoi, f'{self.aoi}_stats{prct2retain}.pkl'))    # stats for the 90% main prodducers
        stats = pd.read_pickle(os.path.join(cst.odir, self.aoi, f'{self.aoi}_stats.pkl'))        # all targetCountry stats
        # drop years before period of interest
        stats = stats.drop(stats[stats['Year'] < project['timeRange'][0]].index)
        # drop unnecessary column
        stats = stats.drop(cst.drop_cols_stats, axis=1)
        # rescale Production units for better graphic and better models
        stats['Production'] = stats['Production'].div(cst.production_scaler)

        # open predictors
        raw_features = pd.read_pickle(os.path.join(cst.odir, self.aoi, f'{self.aoi}_pheno_features4scikit.pkl'))

        # merge stats and features, so that at ech stat entry I have the full set of features (up to year of stats)
        yxData = pd.merge(stats, raw_features, how='left', left_on=['AU_code', 'Year'],
                          right_on=['AU_code', 'YearOfEOS'])

        yxDatac = b1000_preprocess_utilities.retain_X(yxData, statsX, self.crop)
        self.crop_name = yxDatac['Crop_name'].iloc[0]
        self.AUs = list(yxDatac['AU_code'].unique())
        # Get labels (i.e. y values) and ancillary info as separate df
        labels = yxDatac[['Yield', 'Production']]
        years = yxDatac['Year']
        AU_codes = yxDatac['AU_code']

        #############################################
        # retain features based on feature set/method
        #############################################

        if self.model_name == 'PeakNDVI':
            # retain features up to the time of forecast (included) = lead time
            yxDatac = yxDatac.filter(
                regex='|'.join(
                    ['(^|\D)' + self.time_sampling + str(i) + '($|\D)' for i in range(0, self.leadtime + 1)]))
            # ['(^|\D)' + 'M' + str(i) + '($|\D)' for i in range(0, self.leadtime + 1)]))
            # the only feature is max NDVI in the period
            feature_names = ['NDpeak']
            X = yxDatac.filter(like='NDmax').max(axis=1).to_numpy()
            X = X.reshape((-1, 1))
            y = labels[self.yvar].to_numpy()
        elif self.model_name == 'Null_model':
            feature_names = ['None']
            # does not have a feature, is simply the mean by AU of targetVar variable
            y = labels[self.yvar].to_numpy()
            X = y * 0.0
            X = X.reshape((-1, 1))
        elif self.model_name == 'Trend':
            print('Trend model not implemented for forecaster by Michele on 2021-05-21')
            NotImplementedError
        else:  # ML models and Lasso
            X = yxDatac[[x for x in self.selected_features if 'OHE' not in x]].to_numpy()
            y = labels[self.yvar].to_numpy()

            # Preprocessing of input variables using z-score
            if self.model_name not in cst.benchmarks:
                if cst.dataScaling == 'z_f':
                    # scale all features
                    X = self.scaler.fit_transform(X)
                else:
                    NotImplementedError

            # Perform One-Hot Encoding for AU if requested
            if self.doOHE == 'AU_level':
                OHE = pd.get_dummies(AU_codes, columns=['AU_code'], prefix='OHE_AU')
                X = np.concatenate((X, OHE.to_numpy()), axis=1)
                OHE['AU_code'] = AU_codes
                self.OHE = OHE.drop_duplicates()
            else:
                NotImplementedError

        # End of pre-processing of input data -------------------------------------------

        # We use the year as a group (for leave one group out at a time)
        groups = years.to_numpy()
        return X, y, groups, AU_codes

    def fit(self, X, y, years, regions):
        if self.model_name == 'PeakNDVI':
            models = {}
            for au in np.unique(regions):
                index = np.where(regions == au)
                X_au = X[index]
                y_au = y[index]
                reg = linear_model.LinearRegression().fit(X_au, y_au)
                models[au] = reg
            self.search = models
        else:
            cv = LeaveOneGroupOut()
            self.search = b110_modelApp_with_feature_selection.setHyper(
                self.optimisation, self.model_name, self.hyperparams, cv, cst.nJobsForGridSearchCv, self.metric
            )
            # Tune hyperparameters
            self.search.fit(X, y, groups=years)


    def preprocess_currentyear(self, fn, current_year):

        xData = pd.read_csv(fn)  # all targetCountry stats
        xData = xData[xData.YearOfEOS == current_year]
        if self.model_name != 'PeakNDVI':
            xData = xData[xData['AU_code'].isin(self.OHE['AU_code'].unique())]
            xDatac = pd.merge(xData, self.OHE, how='left', on='AU_code', suffixes=('', '_y'))
        else:
            xDatac = xData[xData['AU_code'].isin(self.AUs)]

        #############################################
        # retain features based on feature set/method
        #############################################
        if self.model_name == 'PeakNDVI':
            # retain features up to the time of forecast (included) = lead time
            Xc = xDatac.filter(
                regex='|'.join(
                    ['(^|\D)' + self.time_sampling + str(i) + '($|\D)' for i in range(0, self.leadtime + 1)]))
            # ['(^|\D)' + 'M' + str(i) + '($|\D)' for i in range(0, self.leadtime + 1)]))
            # the only feature is max NDVI in the period
            Xc = Xc.filter(like='NDmax').max(axis=1).to_numpy()
            X = Xc.reshape((-1, 1))

        elif self.model_name == 'Null_model':
            NotImplemented
        else:  # ML models and Lasso
            X = xDatac[[x for x in self.selected_features if 'OHE' not in x]].to_numpy()

            # Preprocessing of input variables using z-score
            if cst.dataScaling == 'z_f':
                # scale all features
                X = self.scaler.transform(X)
            else:
                NotImplementedError

            # Perform One-Hot Encoding for AU if requested
            if self.doOHE == 'AU_level':
                X_OHE = xDatac[[x for x in list(xDatac.columns) if 'OHE' in x]].to_numpy()
                X = np.concatenate((X, X_OHE), axis=1)
            else:
                NotImplemented

        return X, xDatac['AU_code'].to_numpy()


    def predict(self, X_forecast, regions_forecast):
        if self.model_name == 'PeakNDVI':
            y_forecast = np.zeros_like(regions_forecast).astype(np.float)
            for au in np.unique(regions_forecast):
                index = np.where(regions_forecast == au)
                X_au = X_forecast[index]
                y_forecast[index] = self.search[au].predict(X_au)

        elif self.model_name == 'Null_model':
            NotImplementedError
        else:
            y_forecast = self.search.predict(X_forecast)
        return y_forecast


    def predict_uncertainty(self, X, y, groups, regions, X_forecast, regions_forecast):
        preds1 = np.zeros(shape=(np.unique(regions_forecast).shape[0], np.unique(groups).shape[0])).astype(np.float)
        preds2 = np.zeros_like(y).astype(np.float)

        # Bootstrap sampling by leaving one year out
        loyo = LeaveOneGroupOut()
        loyo_gen = loyo.split(X, y, groups=groups)
        cnt = 0
        for train_index, test_index in loyo_gen:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if self.model_name == 'Null_model':
                NotImplemented
            elif self.model_name == 'PeakNDVI':
                regions_train, regions_test = regions.values[train_index], regions.values[test_index]
                preds1_au = np.zeros_like(regions_test).astype(np.float)
                preds2_au = np.zeros_like(regions_test).astype(np.float)
                for au in np.unique(regions_train):
                    X_train_au = X_train[np.where(regions_train == au)]
                    y_train_au = y_train[np.where(regions_train == au)]
                    X_forecast_au = X_forecast[np.where(regions_forecast == au)]
                    X_test_au = X_test[np.where(regions_test == au)]
                    reg = linear_model.LinearRegression().fit(X_train_au, y_train_au)
                    preds2_au[np.where(regions_test == au)] = reg.predict(X_test_au)
                    preds1_au[np.where(regions_forecast == au)] = reg.predict(X_forecast_au)
                # get uncertainty
                preds1[:, cnt] = preds1_au
                cnt += 1
                # get historical performance
                preds2[test_index] = np.abs(preds2_au - y_test)
            else:
                # Train one model per fold
                if self.model_name == 'Lasso':
                    reg = Lasso(alpha=self.search.best_params_['alpha'])
                elif self.model_name == 'RandomForest':
                    reg = RandomForestRegressor(random_state=0, max_depth=self.search.best_params_['max_depth'],
                                                max_features=self.search.best_params_['max_features'],
                                                n_estimators=self.search.best_params_['n_estimators'],
                                                min_samples_split=self.search.best_params_['min_samples_split'])

                elif self.model_name == 'MLP':
                    reg = MLPRegressor(random_state=0, max_iter=100, tol=0.0015,
                                       alpha=self.search.best_params_['alpha'],
                                       hidden_layer_sizes=self.search.best_params_['hidden_layer_sizes'],
                                       activation=self.search.best_params_['activation'],
                                       learning_rate=self.search.best_params_['learning_rate'])

                elif self.model_name == 'GBR':
                    # Gradient Boosting for regression
                    reg = GradientBoostingRegressor(learning_rate=self.search.best_params_['learning_rate'],
                                                    max_depth=self.search.best_params_['max_depth'],
                                                    n_estimators=self.search.best_params_['n_estimators'],
                                                    min_samples_split=self.search.best_params_['min_samples_split'])

                elif self.model_name == 'SVR_linear':  # can be SVR_linear, SVR_rbf
                    reg = SVR(kernel='rbf', C=self.search.best_params_['C'],
                              gamma=self.search.best_params_['gamma'], epsilon=self.search.best_params_['epsilon'])

                elif self.model_name == 'SVR_rbf':  # can be SVR_linear, SVR_rbf
                    reg = SVR(kernel='linear', C=self.search.best_params_['C'],
                              gamma=self.search.best_params_['gamma'], epsilon=self.search.best_params_['epsilon'])
                # fit model
                reg.fit(X_train, y_train)
                # get uncertainty
                preds1[:, cnt] = reg.predict(X_forecast)
                cnt += 1
                # get historical performance
                preds2[test_index] = np.abs(reg.predict(X_test) - y_test)

        uncrt = preds1
        ae = np.reshape(preds2, [y_test.shape[0], -1])

        return uncrt.std(axis=1), ae.mean(axis=1)

    def to_csv(self, regions, forecasts, funcertainty, fmae):
        df_forecast = pd.DataFrame({'ASAP1_ID': np.nan,
                                    'Region_ID': regions,
                                    'Region_name': np.nan,
                                    'Crop_name': self.crop_name,
                                    'fyield_tha': forecasts,
                                    'uncertainty_pct': funcertainty,
                                    'cv_mae': fmae,
                                    'y_percentile': np.nan,
                                    'av_yield_tha': np.nan,
                                    'min_yield_tha': np.nan,
                                    'max_yield_tha': np.nan,
                                    'yield_diff_pct': np.nan,
                                    'farea_ha': np.nan,
                                    'fproduction_t': np.nan,
                                    'p_percentile': np.nan,
                                    'algorithm': self.model_name})

        stats = pd.read_pickle(os.path.join(cst.odir, self.aoi, f'{self.aoi}_stats.pkl'))
        stats = stats[stats['Crop_ID'] == self.crop]

        for region in regions:
            stats_region = stats[stats['Region_ID'] == region]
            yield_region = df_forecast.loc[df_forecast['Region_ID'] == region, 'fyield_tha'].values
            df_forecast.loc[df_forecast['Region_ID'] == region, 'ASAP1_ID'] = stats_region.iloc[0]['ASAP1_ID']
            df_forecast.loc[df_forecast['Region_ID'] == region, 'y_percentile'] = \
                percentile_below(stats_region['Yield'], yield_region)
            df_forecast.loc[df_forecast['Region_ID'] == region, 'Region_name'] = \
                stats_region.iloc[0]['AU_name']
            df_forecast.loc[df_forecast['Region_ID'] == region, 'fproduction_t'] = \
                df_forecast.loc[df_forecast['Region_ID'] == region, 'fyield_tha'] * stats_region['Area'].mean()
            df_forecast.loc[df_forecast['Region_ID'] == region, 'av_yield_tha'] = stats_region['Yield'].mean()
            df_forecast.loc[df_forecast['Region_ID'] == region, 'min_yield_tha'] = stats_region['Yield'].min()
            df_forecast.loc[df_forecast['Region_ID'] == region, 'max_yield_tha'] = stats_region['Yield'].max()
            df_forecast.loc[df_forecast['Region_ID'] == region, 'yield_diff_pct'] = \
                100 * (yield_region - stats_region['Yield'].mean()) / stats_region['Yield'].mean()
            df_forecast.loc[df_forecast['Region_ID'] == region, 'farea_ha'] = stats_region['Area'].mean()
            df_forecast.loc[df_forecast['Region_ID'] == region, 'p_percentile'] = \
                percentile_below(stats_region['Production'],
                                 df_forecast.loc[df_forecast['Region_ID'] == region, 'fproduction_t'].values)

        forecast_fn = os.path.join(
            self.output_dir,
            f'{self.id}_{self.leadtime}_{stats["Crop_name"].unique()[0]}'
            f'_nrt-forecasts_{self.model_name.replace("_", "")}.csv'
        )
        df_forecast.to_csv(forecast_fn, float_format='%.2f')

    def __str__(self):
        return f'------------------------\n' \
               f'The details of the pipeline are: \n' \
               f'Model run ID           : {self.id}\n' \
               f'Area of interest       : {self.aoi}\n' \
               f'Crop type              : {self.crop}\n' \
               f'Algorithm              : {self.model_name}\n' \
               f'Dependent variable     : {self.yvar}\n' \
               f'One hot Encoding       : {self.doOHE}\n' \
               f'Lead time of forecast  : {self.leadtime}\n' \
               f'Time for sampling      : {self.time_sampling} \n' \
               f'Metric to optimise     : {self.metric} \n' \
               f'' \
               f'------------------------'

def percentile_below(x, xt):
    """
    Retrieves the percentile at which a target value is observed
    :param x: distribution
    :param xt: target value
    :return: percentile value at which xt is observed
    """
    if xt > x.max():
        return 1
    p = np.arange(0, 101, 1)
    v = np.percentile(x, p)
    idx = np.where(xt < v)[0][0]
    return p[idx]/100

