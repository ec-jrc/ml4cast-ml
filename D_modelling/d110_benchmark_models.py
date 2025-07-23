import pandas as pd
import numpy as np
from sklearn import linear_model
from tabpfn import TabPFNRegressor


def run_LOYO(model, X_train, X_test, y_train, y_test, adm_id_train, adm_id_test, groups_test):
    # run benchmarks with no INNER LOOP
    yloo_pred = []
    yloo_true = []
    yloo_au = []
    if model == 'Null_model':
        # the prediction for the left out year is the mean of all other year by AU
        for au in adm_id_test:
            # mean of train
            index = np.where(adm_id_train == au)
            y_train_au = y_train[index]
            # treat nan in y
            nas = np.isnan(y_train_au)
            yloo_pred.extend([np.nanmean(y_train_au[~nas])])
            # test
            index = np.where(adm_id_test == au)
            yloo_au.extend(adm_id_test[index].tolist())
            yloo_true.extend(y_test[index].tolist())
        outLoopRes = [yloo_pred, yloo_true, yloo_au, np.unique(groups_test).tolist() * len(yloo_pred)]
    elif model == 'Trend':
        #the prediction for the left out year is precomputed trend
        for au in adm_id_test:
            # the trend for the left out is store in the X
            index = np.where(adm_id_test == au)
            if X_test.shape == (1, 1):
                yloo_pred.append(np.squeeze(X_test)[()])
            else:
                yloo_pred.extend(np.squeeze(X_test)[index])
            yloo_au.extend(adm_id_test[index].tolist())
            yloo_true.extend(y_test[index].tolist())
            outLoopRes = [yloo_pred, yloo_true, yloo_au, np.unique(groups_test).tolist() * len(yloo_pred)]
    elif model == 'PeakNDVI':
        # for each AU tune the model on train and predict on test
        for au in adm_id_test:
            index = np.where(adm_id_train == au)
            X_train_au, y_train_au = X_train[index], y_train[index]
            index = np.where(adm_id_test == au)
            X_test_au, y_test_au = X_test[index], y_test[index]
            # treat nan in y
            nas = np.isnan(y_train_au)
            try:
                reg = linear_model.LinearRegression().fit(X_train_au[~nas].reshape(-1, 1), y_train_au[~nas])
            except:
                print('error d110_benchmark on lin reg PeakNDVI')
                print('Check out why')
            yloo_pred.extend(reg.predict(X_test_au.reshape(-1, 1)).tolist())
            yloo_true.extend(y_test_au.tolist())
            yloo_au.extend(adm_id_test[index].tolist())
            outLoopRes = [yloo_pred, yloo_true, yloo_au, np.unique(groups_test).tolist() * len(yloo_pred)]
    # Tab change 2025
    elif model == 'Tab':
        # reg = TabPFNRegressor()
        # reg.fit(X_train, y_train) #fit
        # y_pred = reg.predict(X_test) #predicts
        # print(y_pred)

        Xdf_train = pd.DataFrame(X_train)
        # test with pd and last as categorial
        # Specify the data type of each column
        for i in range(len(Xdf_train.columns) - 1):
            Xdf_train.iloc[:, i] = Xdf_train.iloc[:, i].astype('float64')
        last_column_name = Xdf_train.columns[-1]
        Xdf_train[last_column_name] = 'label_' + Xdf_train[last_column_name].astype('str')
        Xdf_train[last_column_name] = Xdf_train[last_column_name].astype('category')
        reg = TabPFNRegressor()
        # reg = TabPFNRegressor(categorical_features_indices=[last_column_name])
        reg.fit(Xdf_train, y_train)
        Xdf_test = pd.DataFrame(X_test)
        # test with pd and last as categorial
        # Specify the data type of each column
        for i in range(len(Xdf_test.columns) - 1):
            Xdf_test.iloc[:, i] = Xdf_test.iloc[:, i].astype('float64')
        last_column_name = Xdf_train.columns[-1]
        Xdf_test[last_column_name] = 'label_' + Xdf_test[last_column_name].astype('str')
        Xdf_test[last_column_name] = Xdf_test[last_column_name].astype('category')
        y_pred = reg.predict(Xdf_test)
        # !!! make sure to adjust run_fit as well
        outLoopRes = [y_pred, y_test, adm_id_test, np.unique(groups_test).tolist() * len(y_pred)]

    return outLoopRes

def run_fit(model, X, y, adm_ids):
    # run benchmarks in fitting
    if model == 'PeakNDVI':
        search_list = []
        uniqueadm_id = np.unique(adm_ids)
        y_pred = []
        y_true = []
        for au in uniqueadm_id:
            index = np.where(adm_ids == au)
            X_au, y_au = X[index], y[index]
            # treat nan in y
            nas = np.isnan(y_au)
            reg = linear_model.LinearRegression().fit(X_au[~nas].reshape(-1, 1), y_au[~nas])
            search_list.append(reg)
            y_pred.extend(reg.predict(X_au.reshape(-1, 1)).tolist())
            y_true.extend(y_au.tolist())
    elif model == 'Tab':
        # X is a numpy array
        # reg.fit(X, y)
        # y_true = y
        # y_pred = reg.predict(X)
        Xdf = pd.DataFrame(X)
        # test with pd and last as categorial
        # Specify the data type of each column
        for i in range(len(Xdf.columns) - 1):
            Xdf.iloc[:, i] = Xdf.iloc[:, i].astype('float64')
        last_column_name = Xdf.columns[-1]
        Xdf[last_column_name] = 'label_' + Xdf[last_column_name].astype('str')
        Xdf[last_column_name] = Xdf[last_column_name].astype('category')
        reg = TabPFNRegressor()
        # reg = TabPFNRegressor(categorical_features_indices=[last_column_name])
        reg.fit(Xdf, y)
        y_true = y
        y_pred = reg.predict(Xdf)
        search_list = reg
    return y_true, y_pred, search_list