import time
import os
import json

import pandas as pd

from D_modelling import d100_modeller

# to avoid all warnings (GPR was triggering ConvergenceWarnings)
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"


def fit_and_validate_single_model(fn, config, runType, run2get_mres_only=False):
    # when called with run2get_mres_only = True it just returns mres (for plotting)
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
    if run2get_mres_only:
        #see if mRes there already, in case read, pass back and return
        fn_mRes_out = os.path.join(config.models_out_dir, 'ID_' + str(myID) +
                                   '_crop_' + uset['crop'] + '_Yield_' + uset['algorithm'] +
                                   '_mres.csv')
        if os.path.exists(fn_mRes_out):
            mRes = pd.read_csv(fn_mRes_out)
            return mRes

    if not os.path.exists(fn_out) or run2get_mres_only:
        hindcaster = d100_modeller.YieldModeller(uset)
        # preprocess
        X, y, groups, feature_names, adm_ids = hindcaster.preprocess(config, runType, run2get_mres_only)
        # fit and put results in a dict
        hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, prctPegged, \
        selected_features_names, prct_selected, n_selected, \
        avg_scoring_metric_on_val, fitted_model = hindcaster.fit(X, y, groups, feature_names, adm_ids, runType)
        #if I am retuning or tuning produce MRes for results analysis:
        # Tab change 2025
        # do it for tab as well, it is not re-run in tuning
        if runType == 'tuning' or (uset['algorithm'] == 'Tab' and runType == 'fast_tuning'):
            # write mres for future use
            fn_mRes_out = os.path.join(config.models_out_dir, 'ID_' + str(myID) +
                                  '_crop_' + uset['crop'] + '_Yield_' + uset['algorithm'] +
                                  '_mres.csv')
            mRes.to_csv(os.path.join(fn_mRes_out), index=False)
            # if having mRes was the only purpose, return it back and avoid validation
            if run2get_mres_only:
                return mRes
        runTimeH = (time.time() - tic) / (60 * 60)
        # print(f'Model fitted in {runTimeH} hours')
        # error stats
        hindcaster.validate(hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, prctPegged, runTimeH, feature_names,
                            selected_features_names,
                            prct_selected, n_selected, avg_scoring_metric_on_val, config, runType, save_file=True, save_figs=False)
    else:
        print(myID + ' output files already exist')