import pandas as pd
import os, glob, copy
import numpy as np
from autorank import autorank, plot_stats, create_report, latex_table

import src.constants as cst
from viz.plots import *

def relative_root_mean_square_error(y_true, y_pred):
    '''
    Compute Root Mean Square Percentage Error between two arrays.
    '''
    m_y_true = np.mean(y_true, axis=0)
    loss = np.sqrt(np.mean(np.square(((y_true - y_pred) / m_y_true)), axis=0))

    return loss

def root_mean_square_error(y_true, y_pred):
    '''
    Compute Root Mean Square Percentage Error between two arrays.
    '''
    loss = np.sqrt(np.mean(np.square(((y_true - y_pred))), axis=0))

    return loss

def get_error(g, loss='rrmse'):
    if loss == 'rmse':
        _val = root_mean_square_error(g['yLoo_true'], g['yLoo_pred'])
    elif loss == 'rrmse':
        _val = relative_root_mean_square_error(g['yLoo_true'], g['yLoo_pred'])
    else:
        NotImplemented
    return pd.Series(dict(loss=_val))

def Reverse(tuples):
    new_tup = tuples[::-1]
    return new_tup

def get_from_posterior_matrix(posterior_matrix, idx, idy):
    """

    :param posterior_matrix:
    The value of the matrix in the i-th row and the j-th column contains a 3-tuple (p_smaller, p_equal, p_greater)
    such that p_smaller is the probability that the population in column j is smaller than the population in row i,
    p_equal that both populations are equal, and p_larger that population j is larger than population i.
    :param idx:
    :param idy:
    :return:
    """
    out = [posterior_matrix.loc[idx, idy], posterior_matrix.loc[idy, idx]]
    if isinstance(out[0], tuple):
        out[0] = Reverse(out[0])
    out = [x for x in out if isinstance(x, tuple)]
    if len(out) == 0:
        out = (1., 1., 1.)
    else:
        out = out[0]
    return out

def reformat_model_names(l):
    ''''''
    for n, i in enumerate(l):
        if i == 'Null_model':
            l[n] = 'Null'
        if i == 'RandomForest':
            l[n] = 'RF'
        if i == 'SVR_linear':
            l[n] = 'SVR lin'
        if i == 'SVR_rbf':
            l[n] = 'SVR rbf'
    return l

def create_mat_and_lookup(df_):
    """

    :param df_:
    :return:
    """
    mat_ = np.zeros(df_.shape)

    dic_ = {}
    cnt = 0
    for i in range(mat_.shape[0]):
        for j in range(mat_.shape[1]):
            if df_.iloc[i, j] not in dic_.keys():
                dic_[df_.iloc[i, j]] = cnt
                mat_[i, j] = cnt
                cnt += 1
            else:
                mat_[i, j] = dic_[df_.iloc[i, j]]
    return mat_, dic_

dir_res= 'M:/condor/ML1_data_output/Algeria/Model/'
fn_sum = 'C:/Users/waldnfr/Documents/best_conf_of_all_models.csv'
df_sum = pd.read_csv(fn_sum)
df_sum['forecast_time'] = df_sum['lead_time']
# output columns
model_names = [k for k in cst.hyperGrid.keys()]
model_names.insert(0, 'Null_model')
model_names.insert(1, 'PeakNDVI')

column_names = copy.deepcopy(model_names)
column_names.insert(0, 'Year')

# loop through crop type
for crop_id in df_sum['Crop'].unique():
    # store best method per lead time and its accuracy
    df_acc_x = pd.DataFrame({'forecast_time': df_sum['forecast_time'].unique(),
                               'Method': np.nan,
                               'RMSPE': np.nan})
    df_pst_x = pd.DataFrame(np.zeros([len(model_names), len(df_sum['forecast_time'].unique())]),
                            columns=df_sum['forecast_time'].unique(), index=model_names)
    df_dec_x = pd.DataFrame(np.zeros([len(model_names), len(df_sum['forecast_time'].unique())]),
                            columns=df_sum['forecast_time'].unique(), index=model_names)
    # loop through lead time
    for forecast_time in df_sum['forecast_time'].unique():
        print(f'Crop: {crop_id}; Lead time: {forecast_time}')
        df_sum_xlt = df_sum[(df_sum['forecast_time'] == forecast_time) & (df_sum['Crop'] == crop_id)]

        df_out_xlt = pd.DataFrame({'Year':range(2002, 2018+1)})

        for run_id in df_sum_xlt['runID'].unique():
            run_date = run_id.split('_')[0]
            fn_res = glob.glob(os.path.join(dir_res, run_date, f'*{run_id}*_mRes.csv'))[0]
            model = [x for x in column_names if x in fn_res][0]

            df_res = pd.read_csv(fn_res)\
                .groupby('Year')\
                .apply(get_error)\
                .reset_index()\
                .rename(columns={'loss': model})
            df_out_xlt = df_out_xlt.merge(df_res, on='Year', how='left')
        # Get average RMSPE
        df_acc_xlt = df_out_xlt.mean()\
            .drop('Year')
        best_method = df_acc_xlt.idxmin(axis=0, skipna=True)
        df_acc_x.loc[df_acc_x['forecast_time'] == forecast_time, 'Method'] = best_method
        df_acc_x.loc[df_acc_x['forecast_time'] == forecast_time, 'RMSPE'] = round(df_acc_xlt.min()*100)

        # Prepare data for Baysian test
        df_test = df_out_xlt.drop(columns=['Year'])
        result_bayesian = autorank(df_test, alpha=0.10, verbose=False, approach='bayesian', rope=0.05)

        # Matrix with the pair-wise posterior probabilities estimated with the Bayesian signed ranked test. The matrix
        # is a square matrix with the populations sorted by their central tendencies as rows and columns. The value of
        # the matrix in the i-th row and the j-th column contains a 3-tuple (p_smaller, p_equal, p_greater) such that
        # p_smaller is the probability that the population in column j is smaller than the population in row i, p_equal
        # that both populations are equal, and p_larger that population j is larger than population i. If rope==0.0, the
        # matrix contains only 2-tuples (p_smaller, p_greater) because equality is not possible without a ROPE.
        df_pst_x.iloc[:, forecast_time-1] = pd.Series(
            [get_from_posterior_matrix(result_bayesian.posterior_matrix, best_method, m) for m in model_names],
            index=model_names)

        # The value of the matrix in the i-th row and the j-th column contains the value 'smaller' if the population
        # in column j is significantly larger than the population in row i, 'equal' is both populations are equivalent
        # (i.e., have no practically relevant difference), 'larger' if the population in column j is larger than the
        # population in column i, and 'inconclusive' if the statistical analysis is did not yield a definitive result.
        df_dec_x.iloc[:, forecast_time-1] = pd.Series(
            [str(result_bayesian.decision_matrix.loc[best_method, m]) for m in model_names],
            index=model_names)

        # Format data for plot
    # convert Triplets to Hex codes
    df_ps = df_pst_x.applymap(rgb2hex)
    # Create unique matrix and look up table
    mat_ps, val_dic = create_mat_and_lookup(df_ps)

    # Format data SUBFIG 1
    data1 = np.array([list(np.round(df_acc_x.RMSPE.values, 2))])
    xlabels1 = reformat_model_names(list(df_acc_x.Method.values))
    # Get colormap for SUBFIG 1
    cmap1 = plt.get_cmap('gray')
    cmap1 = truncate_colormap(cmap1, 0.1, 0.95)


    data2 = mat_ps
    ylabels2 = reformat_model_names(list(df_pst_x.index))
    xlabels2 = list(df_pst_x.columns)


    cmap2 = colors.ListedColormap([*val_dic.keys()])
    annot2 =df_dec_x.replace(['inconclusive', 'larger', 'smaller', 'nan', 'equal'], ['I', 'L', 'S', '', 'E'])
    pretty_dates = {1: 'Dec', 2: 'Jan', 3: 'Feb', 4: 'Mar', 5: 'Apr', 6: 'May', 7: 'Jun', 8: 'Jul'}
    xlabels2 = [pretty_dates[j] for j in xlabels2]
    plot_practical_significance(data1, data2, annot2, xlabels1, xlabels2, ylabels2, cmap1=cmap1, cmap2=cmap2, rope=0.05,
                                alpha=0.1, plot_title=crop_id, filename=f'./figures/significance_test_{crop_id}.png')
