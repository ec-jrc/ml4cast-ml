import sys
import time, pickle
import glob
import os

import src.constants as cst
import ml.modeller as modeller
# import warnings
# warnings.filterwarnings("ignore")


#pckl_fn = glob.glob(os.path.join(cst.odir, "Algeria/pkls/20201218", "*"))[80]  # SVM RBF
#pckl_fn = glob.glob(os.path.join(cst.odir, "Algeria/pkls/20201218", "*"))[1080]  # GBR
#pckl_fn = glob.glob(os.path.join(cst.odir, "Algeria/pkls/20201218", "*"))[1160]  # RF
#pckl_fn = glob.glob(os.path.join(cst.odir, "Algeria/pkls/20201218", "*"))[1200]  # SVM linear
#pckl_fn = glob.glob(os.path.join(cst.odir, "Algeria/pkls/20201218", "*"))[150]  # MLP
#pckl_fn = glob.glob(os.path.join(cst.odir, "Algeria/pkls/20201218", "*"))[10]  # MLP

def launcher(pckl_fn):
    # read pickle file
    with open(pckl_fn, 'rb') as f:
        uset = pickle.load(f)
    # Instantiate class for forecasting
    forecaster = modeller.YieldHindcaster(uset['runID'],
                                          uset['target'],
                                          uset['cropID'],
                                          uset['algorithm'],
                                          uset['yvar'],
                                          uset['feature_set'],
                                          uset['feature_selection'],
                                          uset['data_reduction'],
                                          uset['prct_features2select_grid'],
                                          uset['n_features2select_grid'],
                                          uset['doOHE'],
                                          uset['forecast_time'],
                                          uset['yieldTrend'],
                                          uset['time_sampling'])
    print(forecaster)
    if not os.path.exists(os.path.join(forecaster.output_dir_output, f'*{forecaster.id}*_output.csv')):
        X, y, groups, feature_names, AU_codes = forecaster.preprocess(save_to_csv=True)
        tic = time.time()
        hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, nPegged, selected_features_names, \
                    prct_selected, n_selected, avg_scoring_metric_on_val = forecaster.fit(X, y, groups, feature_names, AU_codes, save_output=True)
        runTimeH = (time.time() - tic) / (60 * 60)
        print(f'Model fitted in {runTimeH} hours')

        forecaster.validate(hyperParamsGrid, hyperParams, Fit_R2, coefFit, mRes, nPegged, runTimeH, feature_names, selected_features_names,
                        prct_selected, n_selected, avg_scoring_metric_on_val, save_file=True)
    else:
        print('Output files already exist')


if __name__ == '__main__':
    print(sys.version)
    uset_file = r'{}'.format(sys.argv[1])
    launcher(pckl_fn=uset_file)

