import time
import json
from D_modelling import d100_modeller


def fit_and_validate_single_model(fn, config, runType):
    tic = time.time()
    with open(fn, 'r') as fp:
        uset = json.load(fp)
    print(uset)
    hindcaster = d100_modeller.YieldModeller(uset)
    # preprocess
    X, y, groups, feature_names, AU_codes = hindcaster.preprocess(config, runType)
    # fit and put results in a dict
    hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, prctPegged, \
    selected_features_names, prct_selected, n_selected, \
    avg_scoring_metric_on_val, fitted_model = hindcaster.fit(X, y, groups, feature_names, AU_codes, runType)
    runTimeH = (time.time() - tic) / (60 * 60)
    # print(f'Model fitted in {runTimeH} hours')
    # error stats
    hindcaster.validate(hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, prctPegged, runTimeH, feature_names,
                        selected_features_names,
                        prct_selected, n_selected, avg_scoring_metric_on_val, config, save_file=True, save_figs=False)
