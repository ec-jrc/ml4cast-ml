import pathlib
import pandas as pd
import glob
import os
import sklearn.metrics as metrics
import numpy as np
import pickle
import Model_error_stats

def r2(t, p):
    '''
    r2 of lin reg
    '''
    t_bar = np.mean(t)
    SStot = np.sum(np.square(t-t_bar))
    SSres = np.sum(np.square(t-p))
    r2 = 1.0 - (SSres/SStot)
    return r2
def mean_error(y, x):
    return np.mean(np.array(y) - np.array(x))

def add_error_stats_as_AVG_of_CV_fold(df, dir):
    # stats R2_p	MAE_p	rMAE_p	ME_p	RMSE_p # were computed on all left out together, now we compute as avg of stat by CV folder
    # plus RMSE is wrong
    # I recompute everything here directly fro mRes file
    run_id = df['runID'][0]  # .split('_')[0]
    fn_res = glob.glob(os.path.join(dir, f'*{run_id}*_mRes.csv'))[0]
    df_res = pd.read_csv(fn_res)
    df['R2_p'] = df_res.groupby('Year').apply(lambda x: metrics.r2_score(x['yLoo_true'], x['yLoo_pred'])).reset_index(drop=True).mean()
    df['MAE_p'] = df_res.groupby('Year').apply(lambda x: metrics.mean_absolute_error(x['yLoo_pred'], x['yLoo_true'])).reset_index(drop=True).mean()
    avg_y_true = df_res['yLoo_true'].mean()
    df['rMAE_p'] = df['MAE_p']/avg_y_true*100.0
    df['ME_p'] = df_res.groupby('Year').apply(lambda x: mean_error(x['yLoo_true'], x['yLoo_pred'])).reset_index(drop=True).mean()
    #df['RMSE_p'] = np.sqrt(df_res.groupby('Year').apply(lambda x: metrics.mean_squared_error(x['yLoo_true'], x['yLoo_pred'])).reset_index(drop=True).mean())
    df['RMSE_p'] = df_res.groupby('Year').apply(lambda x: metrics.mean_squared_error(x['yLoo_true'], x['yLoo_pred'], squared=False)).reset_index(
            drop=True).mean()
    df['rRMSE_p'] = df['RMSE_p']/avg_y_true*100.0
    # avg_AU_R2_p(alias R2_WITHINp) was correct
    # countrystats are correct except Country_RMSE_p that nee to be sqrted
    df['Country_RMSE_p'] = df['Country_RMSE_p'].apply(lambda x: np.sqrt(x))
    df['Country_rRMSE_p'] = df['Country_RMSE_p']/avg_y_true*100.0
    df['Country_FQ_RMSE_p'] = df['Country_FQ_RMSE_p'].apply(lambda x: np.sqrt(x))
    df['Country_FQ_rRMSE_p'] = df['Country_FQ_RMSE_p']/avg_y_true*100.0

    res = Model_error_stats.allStats_country_one_year(df_res, 2008)
    df['Country_2008_RMSE_p'] =res['Pred_RMSE_year']
    df['Country_2008_rRMSE_p'] = res['Pred_rRMSE_year']
    # test
    # tmp3 = df_res.groupby('Year').apply(lambda x: r2(x['yLoo_true'], x['yLoo_pred']))
    return df


def gather_output(dir, isAlgeriaPaperRun=False):
    # There was a bug in writing the stats when we run Algeria in 2021 for the paper, gather_output is fixing it isAlgeriaPaperRun = True
    if isAlgeriaPaperRun:
        print('******************************************************')
        print('Warning: output files will be corrected for the bug present in teh Algerian run of March 2021')
        print('******************************************************')
        correct_output = True
    else:
        correct_output = False
    print('correct is ', correct_output)
    run_res = list(pathlib.Path(dir).glob('ID*_output.csv'))
    print('N files =' +str(len(run_res)))
    print('Missing files are printed (no warning issued if they are files that were supposed to be skipped (ft sel asked on 1 var)')
    print('Warnings issued for all other cases, and list at the end')
    list_unkown = []
    cc = 0
    if len(run_res) > 0:
        for file_obj in run_res:
            #print(file_obj, cc)
            cc = cc + 1
            df = pd.read_csv(file_obj)
            run_id = int(df['runID'][0].split('_')[1])
            date_id = str(df['runID'][0].split('_')[0])
            # adjust stats
            if correct_output:
            # add a column for rRMSE_p
                df.insert(28, 'rRMSE_p', np.NAN)
                # add a column for rRMSE_p FQ_rRMSE_p at country level
                df.insert(36, 'Country_rRMSE_p', np.NAN)
                df.insert(38, 'Country_FQ_rRMSE_p', np.NAN)
                df.insert(39, 'Country_2008_RMSE_p', np.NAN)
                df.insert(40, 'Country_2008_rRMSE_p', np.NAN)
                df_updated = add_error_stats_as_AVG_of_CV_fold(df, dir)
            else:
                df_updated = df
            if file_obj == run_res[0]:
                #it is the first, save with hddr
                df_updated.to_csv(file_obj.parent / 'all_model_output.csv', mode='w', header=True, index=False)
            else:
                #it is not first, withou hdr
                df_updated.to_csv(file_obj.parent / 'all_model_output.csv', mode='a', header=False, index=False)
                # print if something is missing
                if run_id > run_id0:
                    if (run_id != run_id0 + 1):
                        for i in range(run_id0 + 1, run_id):
                            print(date_id + ',' + str(i))
                            # # check that the issue is with ft selection requested on 1 var
                            # # open the pkl
                            # myID = f'{date_id}_{i:06d}'
                            # pckl_fn = 'myID + '_uset.pkl'
                            # with open(pckl_fn, 'rb') as f:
                            #     uset = pickle.load(f)
                            #     if not(uset['feature_selection'] == 'MRMR' and len(uset['prct_features2select_grid']) == 1):
                            #         print('Unkown issue with ' + myID + '_uset.pkl')
                            #         print(uset)
                            #         list_unkown.append(myID + '_uset.pkl')
                            #         #print('debug')
                else:
                    print('Date changed?', date_id, 'new run id', run_id0)
            run_id0 = run_id
    else:
        print('There is no a single output file')

    print('list of unkown')
    print(list_unkown)

    #     if len(run_res) > 1:
    #         #read all the others
    #         for file_obj in run_res[1:]:
    #             #read and append to all_res
    #             next_file_df = pd.read_csv(file_obj)
    #             run_id1 =int(next_file_df['runID'][0].split('_')[1])
    #             date_id1 = str(next_file_df['runID'][0].split('_')[0])
    #             if (run_id1 != run_id0 + 1):
    #                 for i in range(run_id0+1, run_id1):
    #                     print(date_id1 + ',' + str(i))
    #                 #print('Missing from ' + date_id0 +' ' + str(run_id0+1) + ' to ' + date_id1 +' ' +str(run_id1-1))
    #             run_id0 = run_id1
    #             next_file_df.to_csv(run_res[0].parent / 'all_model_output.csv', mode='a', header=False, index=False)
    # else:
    #     print('There is no a single output file')
res = gather_output(cst.folder_b200_gather_output, isAlgeriaPaperRun=False)