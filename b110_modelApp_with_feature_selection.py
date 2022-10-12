import pandas as pd
import numpy as np
import copy

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from sklearn.model_selection import LeaveOneGroupOut, LeavePGroupsOut, GroupShuffleSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LassoCV
from sklearn import linear_model, ensemble
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.base import BaseEstimator, RegressorMixin
# from sklearn.utils.testing import ignore_warnings
# from sklearn.exceptions import ConvergenceWarning


from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# import mifs
import getpass
if getpass.getuser() not in ['mmeroni','waldnfr']:
    from ITMO_FS.filters.multivariate import MRMR, MultivariateFilter
import time

import b105_benchmarkModels
import Model_error_stats
from ml.scorer import *

import sys
import os
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

class MLPWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.01, layer1=10, layer2=10, layer3=10, activation='tanh', learning_rate='adaptive'):

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3

        self.alpha=alpha
        self.activation = activation
        self.learning_rate = learning_rate

    def fit(self, X, y):
        model = MLPRegressor(
            alpha=self.alpha,
            activation=self.activation,
            learning_rate=self.learning_rate,
            hidden_layer_sizes=[self.layer1, self.layer2, self.layer3],
            random_state=0, max_iter=10000
        )
        model.fit(X, y)
        self.model = model
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

#@ignore_warnings(category=ConvergenceWarning)
def select_features_and_opt_hyper(X, y, X_test, n_fetures2select_grid, prct_features2select_grid, featureNames,
                                  groups_train,
                                  optimisation, model, param_grid, inner_cv,
                                  nJobsForGridSearchCv, scoringMetric, n_iter_search):
    selected_features_by_feature_set = []
    search_by_feature_set = []
    X_test_by_feature_set = []
    for n in n_fetures2select_grid:
        # perform feature selection
        indices_nonOHE = [i for i, elem in enumerate(featureNames) if not 'OHE' in elem]
        n_features = len(indices_nonOHE)
        if n == n_features:
            X_train_sf = X
            idx_selected_features = [i for i, elem in enumerate(featureNames) if not 'OHE' in elem]  # all nonOHE
        else:
            indices_OHE = [i for i, elem in enumerate(featureNames) if 'OHE' in elem]
            # get non OHE fetaures
            Z = X[:, np.array(indices_nonOHE)]
            Z = np.concatenate((Z, y.reshape(-1, 1)), axis=1)
            est = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
            Z_discrete = est.fit_transform(Z)  # est.fit(Z), est.transform(Z)
            y_discrete = Z_discrete[:, -1]
            X_nonOHE_discrete = Z_discrete[:, 0:-1]
            filterModel = MultivariateFilter('MRMR', n)
            filterModel.fit(X_nonOHE_discrete, y_discrete)
            idx_selected_features = filterModel.selected_features
            # print(model.selected_features)
            # print(np.array(featureNames)[model.selected_features])
            if len(indices_OHE)>0:
                X_train_sf = np.concatenate(
                (X[:, idx_selected_features], X[:, np.array(indices_OHE)]), axis=1)
            else:
                X_train_sf = X[:, idx_selected_features]

        # tune hyper
        n_features = X_train_sf.shape[1] #used by GPR only
        search = setHyper(optimisation, model, param_grid, inner_cv, nJobsForGridSearchCv, scoringMetric,
                          n_iter_search, n_features=n_features)
        # Tune hyperparameters
        search.fit(X_train_sf, y, groups=groups_train)
        # store results
        if len(indices_OHE)>0:
            selected_features_by_feature_set.append(np.array(featureNames)[idx_selected_features].tolist() + np.array(featureNames)[np.array(indices_OHE)].tolist())
            X_test_sf = np.concatenate((X_test[:, idx_selected_features], X_test[:, np.array(indices_OHE)]), axis=1)
        else:
            selected_features_by_feature_set.append(np.array(featureNames)[idx_selected_features].tolist())
            X_test_sf = X_test[:, idx_selected_features]
        X_test_by_feature_set.append(X_test_sf)
        search_by_feature_set.append(search)

    # find out what feature set performed best based on the desired scoringMetric (in best_score_)
    scores = [elem.best_score_ for elem in search_by_feature_set]
    val, idx = max((val, idx) for (idx, val) in enumerate(scores))  # max becauee it is a negative RMSE
    selected_features_names = selected_features_by_feature_set[idx]
    prct_selected = prct_features2select_grid[idx]
    n_selected = n_fetures2select_grid[idx]
    X_test = X_test_by_feature_set[idx]
    search = search_by_feature_set[idx]

    return selected_features_names, prct_selected, n_selected, X_test, search


def setHyper(optimisation, model, param_grid, inner_cv, nJobsForGridSearchCv, scoringMetric, n_iter_search=100, n_features = 0):
    if optimisation == 'grid':
        if model == 'LassoCV':
            search = LassoCV(cv=inner_cv, random_state=0)  # max_iter
        elif model == 'Lasso':
            search = GridSearchCV(estimator=linear_model.Lasso(random_state=0, max_iter=10000),
                                  param_grid=param_grid,
                                  cv=inner_cv, n_jobs=nJobsForGridSearchCv, scoring=scoringMetric)
            # cv=gen_inner_cv, n_jobs=nJobsForGridSearchCv, scoring=scoringMetric)
        elif model == 'RandomForest':
            search = GridSearchCV(estimator=RandomForestRegressor(random_state=0),
                                  param_grid=param_grid,
                                  cv=inner_cv, n_jobs=nJobsForGridSearchCv, scoring=scoringMetric)
            # search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0), n_iter = 10, param_grid = param_grid_RandomForest, random_state = 0,cv = gen_inner_cv)
        elif model == 'MLP':
            # Multi-Layer Perceptron
            search = GridSearchCV(estimator=MLPRegressor(random_state=0, max_iter=100, tol=0.0015),
                                  param_grid=param_grid,
                                  cv=inner_cv, n_jobs=nJobsForGridSearchCv, scoring=scoringMetric)
        elif model == 'GBR':
            # Gradient Boosting for regression
            search = GridSearchCV(estimator=ensemble.GradientBoostingRegressor(random_state=0),
                                  param_grid=param_grid,
                                  cv=inner_cv, n_jobs=nJobsForGridSearchCv, scoring=scoringMetric)
        elif model[0:3] == 'SVR':  # can be SVR_linear, SVR_rbf
            search = GridSearchCV(SVR(epsilon=0.1, kernel=model[4:], cache_size=1000), param_grid=param_grid,
                                  cv=inner_cv, n_jobs=nJobsForGridSearchCv, verbose=0,
                                  scoring=scoringMetric)  # 8
        elif model == 'GPR1':
            # Gaussian process
            l = 0.5
            l_bounds = [(1e-3, 1e+2)]  # length_scale_bounds
            sigma_n = 0.1
            sigma_n_bounds = (1e-5, 1e1)
            kernel = RBF(length_scale=[l] * n_features, length_scale_bounds=l_bounds * n_features) \
                     + WhiteKernel(noise_level=sigma_n, noise_level_bounds=sigma_n_bounds)

            search = GridSearchCV(estimator=GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True),
                                  param_grid=param_grid,
                                  cv=inner_cv, n_jobs=nJobsForGridSearchCv, scoring=scoringMetric)
        elif model == 'GPR2':
            # Gaussian process
            l = 0.5
            l_bounds = [(1e-3, 1e+2)]  # length_scale_bounds
            sigma_f = 1
            sigma_f_bounds = [(1e-1, 1e2)]  # constant_value_bounds
            sigma_n = 0.1
            sigma_n_bounds = (1e-5, 1e1)
            kernel = ConstantKernel(constant_value=sigma_f, constant_value_bounds=sigma_f_bounds) \
                      * RBF(length_scale=[l] * n_features, length_scale_bounds=l_bounds * n_features) \
                      + WhiteKernel(noise_level=sigma_n, noise_level_bounds=sigma_n_bounds)
            search = GridSearchCV(
                estimator=GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True),
                param_grid=param_grid,
                cv=inner_cv, n_jobs=nJobsForGridSearchCv, scoring=scoringMetric)
        else:
            print('Model name not implemented: ' + model)
            exit()
    else:
        print('Optimisation not implemented: ' + model)
        exit()
    # elif optimisation == 'bayesian':
    #     n_iter_search = 60
    #     if model == 'LassoCV':
    #         search = LassoCV(cv=inner_cv, random_state=0)  # max_iter
    #     elif model == 'Lasso':
    #         search = BayesSearchCV(linear_model.Lasso(random_state=0, max_iter=10000),
    #                                param_grid, n_iter=n_iter_search,
    #                                cv=inner_cv, n_jobs=nJobsForGridSearchCv,
    #                                n_points=nJobsForGridSearchCv,
    #                                optimizer_kwargs = {'base_estimator': 'RF'},
    #                                scoring=scoringMetric)
    #     elif model == 'RandomForest':
    #         search = BayesSearchCV(RandomForestRegressor(random_state=0),
    #                                param_grid, n_iter=n_iter_search,
    #                                cv=inner_cv, n_jobs=nJobsForGridSearchCv,
    #                                n_points=nJobsForGridSearchCv,
    #                                optimizer_kwargs= {'base_estimator': 'RF'},
    #                                scoring=scoringMetric)
    #         # search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0), n_iter = 10, param_grid = param_grid_RandomForest, random_state = 0,cv = gen_inner_cv)
    #     elif model == 'MLP':
    #         # Multi-Layer Perceptron
    #         #search = BayesSearchCV(MLPRegressor(random_state=0, max_iter=10000),
    #         #                       hidden_layer_sizes=[Integer(4, 16), Integer(4, 16), Integer(4, 16)],
    #         #                       n_iter=n_iter_search,
    #         #                       cv=inner_cv, n_jobs=nJobsForGridSearchCv, scoring=scoringMetric)
    #         search = BayesSearchCV(estimator=MLPWrapper(),
    #                                search_spaces=param_grid,
    #                                n_iter=n_iter_search, cv=inner_cv, n_jobs=nJobsForGridSearchCv,
    #                                n_points=nJobsForGridSearchCv,
    #                                optimizer_kwargs={'base_estimator': 'RF'},
    #                                scoring=scoringMetric)
    #
    #     elif model == 'GBR':
    #         # Gradient Boosting for regression
    #         search = BayesSearchCV(ensemble.GradientBoostingRegressor(random_state=0),
    #                                param_grid, n_iter=n_iter_search,
    #                                cv=inner_cv, n_jobs=nJobsForGridSearchCv,
    #                                n_points=nJobsForGridSearchCv,
    #                                optimizer_kwargs = {'base_estimator': 'RF'},
    #                                scoring=scoringMetric)
    #     elif model[0:3] == 'SVR':  # can be SVR_linear, SVR_rbf
    #         search = BayesSearchCV(SVR(kernel=model[4:], cache_size=1000),
    #                                param_grid, n_iter=n_iter_search,
    #                                cv=inner_cv, n_jobs=nJobsForGridSearchCv, verbose=0,
    #                                n_points=nJobsForGridSearchCv,
    #                                optimizer_kwargs={'base_estimator': 'RF'},
    #                                scoring=scoringMetric)
    #     else:
    #         print('Model name not implemented: ' + model)
    #         exit()
    return search

#@ignore_warnings(category=ConvergenceWarning)
def modelApp(X, y, groups, featureNames, au_codes, model, addTargetMeanFeatures, scoringMetric, optimisation,
             param_grid, n_out_inner_loop, nJobsForGridSearchCv, n_iter_search=1,
             feature_selection='none', prct_features2select_grid=0, n_fetures2select_grid=0):

    # tune and test a model with double loop to avoid information leak

    # X: the feature matrix (n_sample, n_feature)
    # y: the target label (n_sample)
    # groups: an array with a group identification number, the year for example (n_sample),
    #         it is used for leaving one full group at a time
    # featureNames: names of all features passed (n_ferature) (list)
    # au_codes: an array with ancillary data, the AU code for example (n_sample)
    # model: the model to be applied (string)
    # addTargetMeanFeatures: option, if true the mean of y_train is added as a feature (boolean) -> uset['AddTargetMeanToFeature']
    # scoringMetric: the loss function to be used (string) -> uset['scoringMetric']
    # optimisation: specify the method of hyper optimization (now grid or random)
    # param_grid: the grid for hypers
    # n_out_inner_loop: number of groups to held in the inner loop (it was tested, does not help to set > 1)
    # nJobsForGridSearchCv: numbers of cores to be used when multi-thread is possible, at least 4
    # n_iter_search: used in RandomSearcg
    # feature_selection: optional, set to True to perform features selection. The fraction of features to
    #                    be retained is treated like an hyper optimized in the inner cv loop
    #                    Current possibilities: MRMR
    # feature_grid: the grid of % features to be searched (it is % of features excluding OHE feature)

    # Df to store predictions
    mRes = pd.DataFrame(columns=['yLoo_pred', 'yLoo_true', 'AU_code', 'Year'])

    if feature_selection == 'PCA':
        from sklearn import decomposition
        # just a test
        #  compute PCA
        pca = decomposition.PCA(n_components=3)
        pca.fit(X)
        Xpca = pca.transform(X)
        feature_selection = 'none'
        print('debug')

    # Outer loop for testing model performaces
    # assign the years as groups (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html#sklearn.model_selection.LeaveOneGroupOut)
    outer_cv = LeaveOneGroupOut()
    gen_outer_cv = outer_cv.split(X, y, groups=groups)
    nIterationOuterLoop = 1
    scoring_metric_on_val = []
    for train_index, test_index in gen_outer_cv:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        groups_train, groups_test = groups[train_index], groups[test_index]
        AU_code_train, AU_code_test = au_codes[train_index], au_codes[test_index]
        # M+ exclude records (year - AU) with missing nan
        # train set
        nas = np.isnan(y_train)
        y_train = y_train[~nas]
        X_train = X_train[~nas,:]
        groups_train = groups_train[~nas]
        AU_code_train= AU_code_train[~nas]
        # test set
        nas = np.isnan(y_test)
        y_test = y_test[~nas]
        X_test = X_test[~nas, :]
        groups_test = groups_test[~nas]
        AU_code_test = AU_code_test[~nas]
        # print('Left out year:' + str(np.unique(groups_test)))
        # print(X_train, y_train, groups_train)
        # print(X_test,  y_test, groups_test)
        if model == 'Null_model':
            # No INNER LOOP for the Null model (no hyperparameters), the prediction for the left out year is the mean of all other year by AU
            outLoopRes = b105_benchmarkModels.Null_model(y_train, y_test, AU_code_train, AU_code_test, groups_test)
            mRes = mRes.append(
                pd.DataFrame(np.array(outLoopRes).T.tolist(), columns=['yLoo_pred', 'yLoo_true', 'AU_code', 'Year']))
        elif model == 'Trend':
            # No INNER LOOP for the Trend model
            outLoopRes = b105_benchmarkModels.Trend_model(X_test, y_test, AU_code_test, groups_test)
            mRes = mRes.append(
                pd.DataFrame(np.array(outLoopRes).T.tolist(), columns=['yLoo_pred', 'yLoo_true', 'AU_code', 'Year']))
        elif model == 'PeakNDVI':
            # No INNER LOOP for the PeakNDVI (no hyperparameters)
            outLoopRes = b105_benchmarkModels.PeakNDVI_model(X_train, X_test, y_train, y_test,
                                                             AU_code_train, AU_code_test, groups_test)
            mRes = mRes.append(
                pd.DataFrame(np.array(outLoopRes).T.tolist(), columns=['yLoo_pred', 'yLoo_true', 'AU_code', 'Year']))
        else:
            # DEFINE THE INNER LOOP, ITS COMMON TO ALL THE OTHER (ML/LASSO) METHODS
            # Normally we use 1 out at a time, this was for testing
            if (n_out_inner_loop == 1):
                inner_cv = LeaveOneGroupOut()
            else:
                # Using LeavePGroupsOut with P = 2 would have comb(16,2) = 120, and 560 with P=2 because
                # all possible combinations of P groups are tested. This is too much computationally intensive.
                # Therefore, we use a random selection of those by using GroupShuffleSplit and setting the number of splits
                # to the double of the total number of years len(np.unique(groups_train)). The total number of years
                # would be the number of splits of the LeaveOneGroupOut()
                inner_cv = GroupShuffleSplit(n_splits=len(np.unique(groups_train)) * 2, test_size=n_out_inner_loop,
                                             random_state=42)

            # gen_inner_cv = inner_cv.split(X_train, y_train, groups=groups_train)
            if addTargetMeanFeatures:
                # add the target mean as feature (overall mean of train)
                # add a column to X_target
                meanTargetAU = np.zeros((X_train.shape[0], 1))
                for au in AU_code_test:
                    # mean of train
                    index = np.where(AU_code_train == au)
                    meanTargetAU[index] = np.mean(y_train[index])
                # features are always scaled, here we scale y values
                scaler = StandardScaler()
                meanTargetAU = scaler.fit_transform(meanTargetAU)
                X_train = np.append(X_train, meanTargetAU, axis=1)
                featureNames.append("Y_train_mean")

                # add the target mean as feature also in test
                # add a column to X_test, the mean must come from Y_train, not test
                meanTargetAU = np.zeros((X_test.shape[0], 1))
                for au in AU_code_test:
                    # mean of train
                    index_train = np.where(AU_code_train == au)
                    index_test = np.where(AU_code_test == au)
                    meanTargetAU[index_test] = np.mean(y_train[index_train])
                    meanTargetAU = scaler.fit_transform(meanTargetAU)
                X_test = np.append(X_test, meanTargetAU, axis=1)
            # perform feature selection with a filtering method
            if feature_selection == 'none':
                #print('n_outer', nIterationOuterLoop)
                n_features = X_train.shape[1]         # used by GPR only
                # #######################################
                # #test
                # l = 0.5
                # l_bounds = [(1e-3, 1e+2)] # length_scale_bounds
                # sigma_f = 1
                # sigma_f_bounds = [(1e-1, 1e2)] #constant_value_bounds
                # sigma_n = 0.1
                # sigma_n_bounds = (1e-5, 1e1)
                # kernel = RBF(length_scale=[l] * n_features, length_scale_bounds=l_bounds * n_features) \
                #          + WhiteKernel(noise_level=sigma_n, noise_level_bounds=sigma_n_bounds)
                # kernel2 = ConstantKernel(constant_value=sigma_f, constant_value_bounds=sigma_f_bounds) \
                #          * RBF(length_scale=[l] * n_features, length_scale_bounds=l_bounds * n_features) \
                #          + WhiteKernel(noise_level=sigma_n, noise_level_bounds=sigma_n_bounds)
                # kernel3 = ConstantKernel(constant_value=sigma_f, constant_value_bounds=sigma_f_bounds) \
                #           * (ConstantKernel(constant_value=sigma_f, constant_value_bounds=sigma_f_bounds) \
                #           + RBF(length_scale=[l] * n_features, length_scale_bounds=l_bounds * n_features)) \
                #           + WhiteKernel(noise_level=sigma_n, noise_level_bounds=sigma_n_bounds)
                #
                #
                # rst = [10]
                # print('without constant, y norm')
                # for n_restarts_optimizer in rst:
                #     gp = GaussianProcessRegressor(
                #         kernel=kernel, n_restarts_optimizer=n_restarts_optimizer,
                #         random_state=0, normalize_y=True).fit(X_train, y_train) #(y_train-np.mean(y_train))/np.std(y_train))
                #     lml = gp.log_marginal_likelihood(gp.kernel_.theta)
                #     print(n_restarts_optimizer, lml, 'kerneltheta:', gp.kernel_)
                #     pred, sigma = gp.predict(X_train, return_std=True)
                # print('with constant')
                # for n_restarts_optimizer in rst:
                #     gp2 = GaussianProcessRegressor(
                #         kernel=kernel2, n_restarts_optimizer=n_restarts_optimizer,
                #         random_state=0, normalize_y=True, alpha=sigma_n**2).fit(X_train, y_train) # (y_train-np.mean(y_train))/np.std(y_train))
                #     lml = gp2.log_marginal_likelihood(gp2.kernel_.theta)
                #     print(n_restarts_optimizer, lml, 'kerneltheta:', gp2.kernel_)
                #     pred2, sigma2 = gp2.predict(X_train, return_std=True)
                # print('with 2 constant')
                # for n_restarts_optimizer in rst:
                #     gp3 = GaussianProcessRegressor(
                #         kernel=kernel3, n_restarts_optimizer=n_restarts_optimizer,
                #         random_state=0, normalize_y=True).fit(X_train, y_train)  # (y_train-np.mean(y_train))/np.std(y_train))
                #     lml = gp3.log_marginal_likelihood(gp3.kernel_.theta)
                #     print(n_restarts_optimizer, lml, 'kerneltheta:', gp3.kernel_)
                #     pred3, sigma3 = gp3.predict(X_train, return_std=True)
                # #plt.scatter(y_train, pred2)
                # #plt.scatter(y_train, pred)
                # print('R+W y_norm RMSE', metrics.mean_squared_error(y_train, pred, squared=False))
                # print('C*R+W RMSE', metrics.mean_squared_error(y_train, pred2, squared=False))
                # print('C1*(C2+R)+W RMSE', metrics.mean_squared_error(y_train, pred3, squared=False))
                # plt.figure(figsize=(7,30))
                # plt.subplot(3, 2, 1)
                # u, v = pred, pred2
                # plt.scatter(u, v)
                # plt.plot([0,100],[0,100],'black')
                # plt.xlim([np.min([u, v]), np.max([u, v])])
                # plt.ylim([np.min([u, v]), np.max([u, v])])
                # plt.xlabel('R+W y_norm')
                # plt.ylabel('C*R+W')
                #
                # plt.subplot(3, 2, 2)
                # u, v = pred, pred3
                # plt.scatter(u, v)
                # plt.plot([0, 100], [0, 100], 'black')
                # plt.xlim([np.min([u, v]), np.max([u, v])])
                # plt.ylim([np.min([u, v]), np.max([u, v])])
                # plt.xlabel('R+W y_norm')
                # plt.ylabel('C1*(C2+R)+W')
                #
                # plt.subplot(3, 2, 3)
                # u, v = y_train, pred
                # plt.scatter(u, v)
                # plt.plot([0, 100], [0, 100], 'black')
                # plt.xlim([np.min([u, v]), np.max([u, v])])
                # plt.ylim([np.min([u, v]), np.max([u, v])])
                # plt.xlabel('y_train')
                # plt.ylabel('R+W y_norm')
                #
                # plt.subplot(3, 2, 4)
                # u, v = y_train, pred2
                # plt.scatter(u, v)
                # plt.plot([0, 100], [0, 100], 'black')
                # plt.xlim([np.min([u, v]), np.max([u, v])])
                # plt.ylim([np.min([u, v]), np.max([u, v])])
                # plt.xlabel('y_train')
                # plt.ylabel('C*R+W')
                #
                # plt.subplot(3, 2, 5)
                # u, v = y_train, pred3
                # plt.scatter(u, v)
                # plt.plot([0, 100], [0, 100], 'black')
                # plt.xlim([np.min([u, v]), np.max([u, v])])
                # plt.ylim([np.min([u, v]), np.max([u, v])])
                # plt.xlabel('y_train')
                # plt.ylabel('C1*(C2+R)+W')
                #
                # plt.show()
                # # end of test
                # #######################################
                search = setHyper(optimisation, model, param_grid, inner_cv, nJobsForGridSearchCv, scoringMetric, n_features = n_features)
                # Tune hyperparameters
                search.fit(X_train, y_train, groups=groups_train)
                # End of INNER LOOP fo ML models
            elif feature_selection == 'MRMR':
                selected_features_names, prct_selected, n_selected, X_test, search = select_features_and_opt_hyper(
                    X_train, y_train, X_test,
                    n_fetures2select_grid, prct_features2select_grid, featureNames, groups_train,
                    optimisation, model, param_grid, inner_cv,
                    nJobsForGridSearchCv, scoringMetric, n_iter_search)
                # End of INNER LOOP fo ML models
            else:
                print('Feature selection type not implemented: ' + model)
                exit()
            # add the scoring metric in validation
            #print(search.best_params_)
            scoring_metric_on_val.append(search.best_score_)
            #  Now predict the left outs with tuned hypers and store the prediction for the left-out (all years)
            outLoopRes = [search.predict(X_test).tolist(), y_test.tolist(), AU_code_test.tolist(),
                          np.unique(groups_test).tolist() * len(AU_code_test.tolist())]
            mRes = mRes.append(
                pd.DataFrame(np.array(outLoopRes).T.tolist(), columns=['yLoo_pred', 'yLoo_true', 'AU_code', 'Year']))
            # check and store if best params are pegged to boundaries
            if nIterationOuterLoop == 1:
                # Define variable to keep track on pegging to grid boundaries
                nPeggedLeft = dict((el, 0) for el in param_grid.keys())
                nPeggedRight = copy.deepcopy(nPeggedLeft)
            keys = list(search.best_params_.keys())
            for k in keys:
                if not isinstance(param_grid[k], Integer):
                    if search.best_params_[k] == param_grid[k][0]:
                        nPeggedLeft[k] = nPeggedLeft[k] + 1
                    if search.best_params_[k] == param_grid[k][-1]:
                        nPeggedRight[k] = nPeggedRight[k] + 1
                else:
                    if search.best_params_[k] == param_grid[k].low:
                        nPeggedLeft[k] = nPeggedLeft[k] + 1
                    if search.best_params_[k] == param_grid[k].high:
                        nPeggedRight[k] = nPeggedRight[k] + 1
        # End of outer loop
        nIterationOuterLoop += 1

    # Fitting stats and summary of nPegged
    nPegged = {'left': '', 'right': ''}
    if (model == 'Null_model'):
        hyperParamsGrid = np.nan
        hyperParams = np.nan
        Fit_R2 = np.nan
        coefFit = np.nan
        selected_features_names, prct_selected, n_selected = '', '', ''
        avg_scoring_metric_on_val = np.nan
    elif (model == 'Trend'):
        hyperParamsGrid = np.nan
        hyperParams = np.nan
        Fit_R2 = np.nan
        coefFit = np.nan
        selected_features_names, prct_selected, n_selected = '', '', ''
        avg_scoring_metric_on_val = np.nan
    elif (model == 'PeakNDVI'):
        hyperParamsGrid = np.nan
        hyperParams = np.nan
        selected_features_names, prct_selected, n_selected = '', '', ''
        uniqueAU_code = np.unique(au_codes)
        yloo_pred = []
        yloo_true = []
        for au in uniqueAU_code:
            index = np.where(au_codes == au)
            # M+
            X_au = X[index]
            y_au = y[index]
            # treat nan in y
            nas = np.isnan(y_au)
            reg = linear_model.LinearRegression().fit(X_au[~nas].reshape(-1, 1), y_au[~nas])
            yloo_pred.extend(reg.predict(X_au.reshape(-1, 1)).tolist())
            yloo_true.extend(y_au.tolist())

        Fit_R2 = Model_error_stats.r2_nan(np.array(yloo_true), np.array(yloo_pred)) #metrics.r2_score(np.array(yloo_true), np.array(yloo_pred))
        coefFit = np.nan
        avg_scoring_metric_on_val = np.nan
    else:
        if addTargetMeanFeatures:
            # add the target mean as feature (overall mean of train)
            # add a column to X_target
            meanTargetAU = np.zeros((X.shape[0], 1))
            for au in np.unique(au_codes):
                index = np.where(au_codes == au)
                meanTargetAU[index] = np.mean(y[index])
            meanTargetAU = scaler.fit_transform(meanTargetAU)
            X = np.append(X, meanTargetAU, axis=1)
        # regenerate an outer loop for setting hyperparameters
        outer_cv = LeaveOneGroupOut()
        # M+
        nas = np.isnan(y)
        y = y[~nas]
        X = X[~nas,:]
        groups = groups[~nas]
        if feature_selection == 'none':
            #search = setHyper(optimisation, model, param_grid, inner_cv, nJobsForGridSearchCv, scoringMetric)
            # MM on 21 12 2021
            n_features = X_train.shape[1]  # used by GPR only
            search = setHyper(optimisation, model, param_grid, outer_cv, nJobsForGridSearchCv, scoringMetric,  n_features=n_features)
            # Tune hyperparameters
            search.fit(X, y, groups=groups)
            Fit_R2 = metrics.r2_score(y, search.predict(X))
            #selected_features_names = featureNames
            selected_features_names, prct_selected, n_selected = '', '', ''
        elif feature_selection == 'MRMR':
            # Tune hyperparameters (including feature selection)
            selected_features_names, prct_selected, n_selected, X_test, search = select_features_and_opt_hyper(X, y, X,
                                                                                                               n_fetures2select_grid,
                                                                                                               prct_features2select_grid,
                                                                                                               featureNames,
                                                                                                               groups,
                                                                                                               optimisation,
                                                                                                               model,
                                                                                                               param_grid,
                                                                                                               outer_cv,
                                                                                                               nJobsForGridSearchCv,
                                                                                                               scoringMetric,
                                                                                                               n_iter_search)

            # search.fit(X, y, groups=groups)
            Fit_R2 = metrics.r2_score(y, search.predict(X_test)) # It is X test because it is X with only the selected features

        if model == 'LassoCV':
            hyperParams = 'alpha' + ':' + str(search.alpha_)
            hyperParamsGrid = 'Automatic'
            coefFit = ';'.join(
                [i + ':' + str(round(j, 3)) for i, j in zip(selected_features_names, search.coef_.tolist())])
        if model == 'Lasso':
            hyperParams = 'alpha' + ':' + str(search.best_params_['alpha'])
            hyperParamsGrid = '; '.join(
                str(key) + ':' + ','.join(str(element) for element in list(value)) for key, value in param_grid.items())
            coefFit = ','.join(
                [i + ':' + str(round(j, 3)) for i, j in
                 zip(selected_features_names, search.best_estimator_.coef_.tolist())])
        if model == 'RandomForest':
            hyperParams = '; '.join(str(key) + ':' + str(value) for key, value in search.best_params_.items())
            hyperParamsGrid = '; '.join(
                str(key) + ':' + ','.join(str(element) for element in list(value)) for key, value in param_grid.items())
            # attempt to get var importance
            coefFit = ','.join([i + ':' + str(round(j, 3)) for i, j in
                                zip(selected_features_names, search.best_estimator_.feature_importances_.tolist())])
        if model == 'MLP':
            hyperParams = '; '.join(str(key) + ':' + str(value) for key, value in search.best_params_.items())
            for key, value in param_grid.items():
                if isinstance(value, Integer):
                    param_grid[key] = ['low='+str(value.low), 'high='+str(value.high)]
            hyperParamsGrid = '; '.join(
                str(key) + ':' + ','.join(str(element) for element in list(value)) for key, value in param_grid.items())
            # attempt to get var importance
            coefFit = np.nan
        if model == 'GBR':
            hyperParams = '; '.join(str(key) + ':' + str(value) for key, value in search.best_params_.items())
            hyperParamsGrid = '; '.join(
                str(key) + ':' + ','.join(str(element) for element in list(value)) for key, value in param_grid.items())
            # attempt to get var importance
            coefFit = ','.join([i + ':' + str(round(j, 3)) for i, j in
                                zip(selected_features_names, search.best_estimator_.feature_importances_.tolist())])
        if model[0:3] == 'SVR':
            hyperParams = '; '.join(str(key) + ':' + str(value) for key, value in search.best_params_.items())
            hyperParamsGrid = '; '.join(
                str(key) + ':' + ','.join(str(element) for element in list(value)) for key, value in param_grid.items())
            coefFit = np.nan
        if model == 'GPR1' or model=='GPR2':
            hyperParams = '; '.join(str(key) + ':' + str(value) for key, value in search.best_params_.items()) + '; ' + str(search.best_estimator_.kernel_)
            hyperParamsGrid = '; '.join(
                str(key) + ':' + ','.join(str(element) for element in list(value)) for key, value in param_grid.items())
            coefFit = np.nan

        nIterationOuterLoop = nIterationOuterLoop - 1  # remove last increment at the the end of loop (the increment is at the end of the loop)
        for (key, value) in nPeggedLeft.items():
            if value > 0:
                nPegged['left'] = nPegged['left'] + key + ':' + str(round(value / nIterationOuterLoop * 100)) + ';'
        for (key, value) in nPeggedRight.items():
            if value > 0:
                nPegged['right'] = nPegged['right'] + key + ':' + str(round(value / nIterationOuterLoop * 100)) + ';'
        avg_scoring_metric_on_val = np.mean([-x for x in scoring_metric_on_val]) # rmse is negative, turn it positive
    return hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, nPegged, selected_features_names, prct_selected, n_selected, avg_scoring_metric_on_val
