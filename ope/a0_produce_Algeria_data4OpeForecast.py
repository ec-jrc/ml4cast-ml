from preprocess import b20_LoadCsv_savePickle, b60_build_features

target = 'Algeria'
# store the updated data in: -PY_data-ML1_data_input-Predictors_4OPE
b20_LoadCsv_savePickle.LoadCsv_savePickle(target, 'Predictors_4OPE', ope_run=True)
b60_build_features.build_features(target, ope_run=True)