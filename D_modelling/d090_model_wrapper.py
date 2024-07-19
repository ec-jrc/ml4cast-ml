import time
import os
import json
from D_modelling import d100_modeller

# to avoid all warnings (GPR was triggering ConvergenceWarnings)
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"


def fit_and_validate_single_model(fn, config, runType, run2get_mres_only=False):
    # when called with run2get_mres_only = True it just return mres (for plotting)
    # and does not write anything
    tic = time.time()
    with open(fn, 'r') as fp:
        uset = json.load(fp)
    #print(uset)
    myID = uset['runID']
    myID = f'{myID:06d}'
    fn_out = os.path.join(config.models_out_dir, 'ID_' + str(myID) +
                          '_crop_' + uset['crop'] + '_Yield_' + uset['algorithm'] +
                          '_output.csv')
    if not os.path.exists(fn_out) or run2get_mres_only:
        hindcaster = d100_modeller.YieldModeller(uset)
        # preprocess
        X, y, groups, feature_names, AU_codes = hindcaster.preprocess(config, runType, run2get_mres_only)
        # fit and put results in a dict
        hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, prctPegged, \
        selected_features_names, prct_selected, n_selected, \
        avg_scoring_metric_on_val, fitted_model = hindcaster.fit(X, y, groups, feature_names, AU_codes, runType)
        if run2get_mres_only:
            return mRes
        runTimeH = (time.time() - tic) / (60 * 60)
        # print(f'Model fitted in {runTimeH} hours')
        # error stats
        hindcaster.validate(hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, prctPegged, runTimeH, feature_names,
                            selected_features_names,
                            prct_selected, n_selected, avg_scoring_metric_on_val, config, save_file=True, save_figs=False)
    else:
        print(myID + ' output files already exist')