from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model, ensemble
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import xgboost as xgb
import numpy as np
import mrmr
import pandas as pd

def setHyper(model, param_grid, inner_cv, nJobsForGridSearchCv, scoringMetric, n_features = 0):
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
    elif model == 'GPR': # Former GPR2
        # Gaussian process
        l = 0.5
        l_bounds = [(1e-3, 1e+2)]  # length_scale_bounds
        # l_bounds = [(1e-3, 1e+5)]  # length_scale_bounds
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
    elif model == 'XGBoost':
        # XGBoost for regression
        search = GridSearchCV(estimator=xgb.XGBRegressor(n_jobs=1, random_state=0), #, early_stopping_rounds=3
                              param_grid=param_grid,
                              cv=inner_cv, n_jobs=nJobsForGridSearchCv, scoring=scoringMetric)
    else:
        print('Model name not implemented: ' + model)
        exit()
    return search

# updated version that does not use ITSMO but https://github.com/smazzanti/mrmr
def setHyper_ft_sel(X, y, X_test, prct_features2select_grid, featureNames,
                                  groups_train,
                                  model, param_grid, inner_cv,
                                  nJobsForGridSearchCv, scoringMetric):
    # X_test is also needed because it has to be changed according to the ft selection
    selected_features_by_feature_set = []
    search_by_feature_set = []
    X_test_by_feature_set = []
    # Use mrmr to sort features in descending importance
    # get features to be tested (exclude OHE and YieldFromTrend)
    indices_OHE = [i for i, elem in enumerate(featureNames) if ('OHE' in elem) or ('Yield') in elem]
    indices_nonOHE = [i for i, elem in enumerate(featureNames) if not ('OHE' in elem or ('Yield') in elem)]
    n_features = len(indices_nonOHE)
    Z = X[:, np.array(indices_nonOHE)]
    Z_ft_names =np.array(featureNames)[indices_nonOHE]
    Z_test = X_test[:, np.array(indices_nonOHE)]
    dfZ = pd.DataFrame(Z)
    dfy = pd.Series(y)

    # the number of features may be reduced because of PCA, I have to updated
    n_features2select_grid = n_features * np.array(prct_features2select_grid) / 100
    n_features2select_grid = np.round(n_features2select_grid, decimals=0, out=None)
    # keep those with at least 1 feature
    idx2retain = [idx for idx, value in enumerate(n_features2select_grid) if value >= 1]
    prct_features2select_grid = np.array(prct_features2select_grid)[idx2retain]
    n_features2select_grid = n_features2select_grid[idx2retain]
    # drop possible duplicate in n, and if the same number of ft referes to multiple %, take the largest (this explain the np.flip)
    n_features2select_grid, idx2retain = np.unique(np.flip(np.array(n_features2select_grid)), return_index=True)
    n_features2select_grid = n_features2select_grid.tolist()
    prct_features2select_grid = np.flip(prct_features2select_grid)[idx2retain].tolist()
    # Note: in case PCA selected, the number of feature is recomputed on selected PCA



    # get ranked idx of variables
    ranked_idx_selected_features = mrmr.mrmr_regression(X=dfZ, y=dfy, K=int(max(n_features2select_grid)), show_progress=True)
    for n in list(map(int, n_features2select_grid)):
        idx_selected_features = sorted(ranked_idx_selected_features[0: n])
        if len(indices_OHE) > 0:
            X_train_sf = np.concatenate((Z[:, idx_selected_features], X[:, np.array(indices_OHE)]), axis=1)
        else:
            X_train_sf = X[:, idx_selected_features]

        # tune hyper
        n_features = X_train_sf.shape[1] #used by GPR only
        search = setHyper(model, param_grid, inner_cv, nJobsForGridSearchCv, scoringMetric, n_features = n_features)
        search.fit(X_train_sf, y, groups=groups_train)
        # store results
        if len(indices_OHE)>0:
            selected_features_by_feature_set.append(Z_ft_names[idx_selected_features].tolist() + np.array(featureNames)[np.array(indices_OHE)].tolist())
            X_test_sf = np.concatenate((Z_test[:, idx_selected_features], X_test[:, np.array(indices_OHE)]), axis=1)
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
    n_selected = n_features2select_grid[idx]
    X_test = X_test_by_feature_set[idx]
    search = search_by_feature_set[idx]

    return selected_features_names, prct_selected, n_selected, X_test, search