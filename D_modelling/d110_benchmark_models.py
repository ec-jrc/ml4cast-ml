import pandas as pd
import numpy as np
from sklearn import linear_model


def run_LOYO(model, X_train, X_test, y_train, y_test, AU_code_train, AU_code_test, groups_test):
    # run benchmarks with no INNER LOOP
    yloo_pred = []
    yloo_true = []
    yloo_au = []
    if model == 'Null_model':
        # the prediction for the left out year is the mean of all other year by AU
        for au in AU_code_test:
            # mean of train
            index = np.where(AU_code_train == au)
            y_train_au = y_train[index]
            # treat nan in y
            nas = np.isnan(y_train_au)
            yloo_pred.extend([np.nanmean(y_train_au[~nas])])
            # test
            index = np.where(AU_code_test == au)
            yloo_au.extend(AU_code_test[index].tolist())
            yloo_true.extend(y_test[index].tolist())
        outLoopRes =  [yloo_pred, yloo_true, yloo_au, np.unique(groups_test).tolist() * len(yloo_pred)]

    elif model == 'Trend':
        #the prediction for the left out year is precomputed trend
        for au in AU_code_test:
            # the trend for the left out is store in the X
            index = np.where(AU_code_test == au)
            yloo_pred.extend(np.squeeze(X_test)[index])
            yloo_au.extend(AU_code_test[index].tolist())
            yloo_true.extend(y_test[index].tolist())
            outLoopRes = [yloo_pred, yloo_true, yloo_au, np.unique(groups_test).tolist() * len(yloo_pred)]
    elif model == 'PeakNDVI':
        # for each AU tune the model on train and predict on test
        for au in AU_code_test:
            index = np.where(AU_code_train == au)
            X_train_au, y_train_au = X_train[index], y_train[index]
            index = np.where(AU_code_test == au)
            X_test_au, y_test_au = X_test[index], y_test[index]
            # treat nan in y
            nas = np.isnan(y_train_au)
            reg = linear_model.LinearRegression().fit(X_train_au[~nas].reshape(-1, 1), y_train_au[~nas])
            yloo_pred.extend(reg.predict(X_test_au.reshape(-1, 1)).tolist())
            yloo_true.extend(y_test_au.tolist())
            yloo_au.extend(AU_code_test[index].tolist())
            outLoopRes = [yloo_pred, yloo_true, yloo_au, np.unique(groups_test).tolist() * len(yloo_pred)]

    return outLoopRes

def run_fit(model, X, y, AU_codes):
    # run benchmarks in fitting
    if model == 'PeakNDVI':
        uniqueAU_code = np.unique(AU_codes)
        y_pred = []
        y_true = []
        for au in uniqueAU_code:
            index = np.where(AU_codes == au)
            X_au, y_au = X[index], y[index]
            # treat nan in y
            nas = np.isnan(y_au)
            reg = linear_model.LinearRegression().fit(X_au[~nas].reshape(-1, 1), y_au[~nas])
            y_pred.extend(reg.predict(X_au.reshape(-1, 1)).tolist())
            y_true.extend(y_au.tolist())
    return y_true, y_pred, reg