import numpy as np

def get(model, search, param_grid, selected_features_names): #,nPeggedLeft, nPeggedRight, nIterationOuterLoop):
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
        coefFit ='Intercept: ' + str(search.best_estimator_.intercept_) + ','+  coefFit
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
            if isinstance(value, int):
                param_grid[key] = ['low=' + str(value.low), 'high=' + str(value.high)]
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
    #if model == 'GPR1' or model == 'GPR2':
    if model == 'GPR':
        hyperParams = '; '.join(str(key) + ':' + str(value) for key, value in search.best_params_.items()) + '; ' + str(
            search.best_estimator_.kernel_)
        hyperParamsGrid = '; '.join(
            str(key) + ':' + ','.join(str(element) for element in list(value)) for key, value in param_grid.items())
        coefFit = np.nan
    if model == 'XGBoost':
        hyperParams = '; '.join(str(key) + ':' + str(value) for key, value in search.best_params_.items())
        hyperParamsGrid = '; '.join(
            str(key) + ':' + ','.join(str(element) for element in list(value)) for key, value in param_grid.items())
        # attempt to get var importance
        coefFit =  np.nan

    return hyperParams, hyperParamsGrid, coefFit