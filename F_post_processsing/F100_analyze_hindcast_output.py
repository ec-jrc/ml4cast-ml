import pathlib
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import numpy as np
import datetime
from string import digits
import ast
import os
from A_config import a10_config

def gather_output(config):
    run_res = list(sorted(pathlib.Path(config.models_out_dir).glob('ID*_output.csv')))
    print('N files = ' + str(len(run_res)))
    print('Missing files are printed ') #(no warning issued if they are files that were supposed to be skipped (ft sel asked on 1 var)')
    cc = 0
    if len(run_res) > 0:
        for file_obj in run_res:
            #print(file_obj, cc)
            cc = cc + 1
            try:
                df = pd.read_csv(file_obj)
            except:
                print('Empty file ' + str(file_obj))
            else:
                try:
                    run_id = int(df['runID'][0]) #.split('_')[1])
                except:
                    print('Error in the file ' + str(file_obj))
                else:
                    # date_id = str(df['runID'][0].split('_')[0])
                    # df_updated = df

                    if file_obj == run_res[0]:
                        #it is the first, save with hdr
                        df.to_csv(file_obj.parent / 'all_model_output.csv', mode='w', header=True, index=False)
                    else:
                        #it is not first, without hdr
                        df.to_csv(file_obj.parent / 'all_model_output.csv', mode='a', header=False, index=False)
                        # print if something is missing
                        if run_id > run_id0:
                            if (run_id != run_id0 + 1):
                                for i in range(run_id0 + 1, run_id):
                                    print('Non consececutive runids:' + str(i))
                        # else:
                        #     print('Date changed?', date_id, 'new run id', run_id0)
                    run_id0 = run_id
    else:
        print('There is no a single output file')

def addStringIfNotEmpty(info, x, sep = ','):
    if x != '':
        if len(info)>0:
            info = info + sep +'  ' + str(x)
        else:
            info = str(x)
    return info

def output_row_to_ML_info_string(df, metric2use):
    info_string0 = round(df[metric2use].values[0],2)
    algo = df.Estimator.values[0]
    info_string1 = algo
    ohe = 'OHEau' if df.DoOHEnc.values[0] == 'AU_level' else ''
    info_string1 = addStringIfNotEmpty(info_string1, ohe)
    yt = 'YT' if df.AddYieldTrend.values[0] == 'True' else ''
    info_string1 = addStringIfNotEmpty(info_string1, yt)
    info_string2 = df.Feature_set.values[0]
    dr = df.Data_reduction.values[0] if df.Data_reduction.values[0] != 'none' else ''
    info_string3 = addStringIfNotEmpty('', dr)
    fs = df.Ft_selection.values[0] if df.Ft_selection.values[0] != 'none' else ''
    info_string3 = addStringIfNotEmpty('', fs)
    prct= round(df.Prct_selected_fit.values[0]) if fs != '' else ''
    info_string3 = addStringIfNotEmpty(info_string3, prct, sep=':')
    return [info_string0, info_string1, info_string2, info_string3]


def compare_outputs (config, metric2use = 'rRMSE_p'):  #RMSE_p' #'R2_p'
    #get some vars from mlSettings class
    mlsettings = a10_config.mlSettings(forecastingMonths=0)
    includeTrendModel = True   #for Algeria run ther is no trend model
    addCECmodel = False #only for South Africa
    ylim_rRMSE_p = [0,30]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 400)  # width is = 400

    if metric2use == 'R2_p':
        sortAscending = False
    elif metric2use == 'RMSE_p':
        sortAscending = True
    elif metric2use == 'RMSE_val':
        sortAscending = True
    elif metric2use == 'rRMSE_p':
        sortAscending = True
    else:
        print('The metric is not coded, compare_outputs cannot be executed')
        sys.exit()
    var4time = 'forecast_time'

    mo = pd.read_csv(config.models_out_dir + '/' + 'all_model_output.csv')
    # get best 4 ML configurations by lead time, crop type and y var PLUS benchmarks
    moML = mo[mo['Estimator'].isin(mlsettings.benchmarks) == False]
    b4 = moML.groupby(['Crop', var4time]).apply(lambda x: x.sort_values([metric2use], ascending = sortAscending).head(4)).reset_index(drop=True)
    # always add the benchmarks
    tmp = mo.groupby(['Crop', var4time]) \
        .apply(lambda x: x.loc[x['Estimator'].isin(mlsettings.benchmarks)]).reset_index(drop=True)
    tmp = tmp.drop_duplicates(subset=[var4time, 'Estimator', 'Crop'])
    b4 = pd.concat([b4, tmp])
    b4 = b4.sort_values([var4time,'Crop', metric2use], \
                   ascending=[True, True, sortAscending])
    b4.to_csv(config.models_out_dir + '/' + 'all_model_best4.csv', index=False)

    # and absolute ML best (plus benchmarks)
    # get best 4 ML configurations by lead time, crop type and y var PLUS benchmarks
    moML = mo[mo['Estimator'].isin(mlsettings.benchmarks) == False]
    b1ML = moML.groupby(['Crop', var4time]).apply(lambda x: x.sort_values([metric2use], ascending = sortAscending).head(1)).reset_index(drop=True)
    # always add the benchmarks
    tmp = mo.groupby(['Crop', var4time]) \
        .apply(lambda x: x.loc[x['Estimator'].isin(mlsettings.benchmarks)]).reset_index(drop=True)
    tmp = tmp.drop_duplicates(subset=[var4time, 'Estimator', 'Crop'])
    b1 = pd.concat([b1ML, tmp])

    # b1 = mo.groupby(['Crop', var4time]).apply(
    #     lambda x: x.sort_values([metric2use], ascending=sortAscending).head(1)).reset_index(drop=True)
    # b1['Ordering'] = 0  # just used to order Ml, peak, null in the csv
    # for idx, val in enumerate(mlsettings.benchmarks, start=0):
    #     tmp = mo.groupby(['Crop', var4time]) \
    #         .apply(lambda x: x.loc[x['Estimator'].isin([mlsettings.benchmarks[idx]])]).reset_index(drop=True)
    #     tmp['Ordering'] = idx+1
    #     tmp = tmp.drop_duplicates(subset=[var4time, 'Estimator', 'Crop'])
    #     b1 = pd.concat([b1, tmp])
    # b1 = b1.sort_values(['Crop', var4time, 'Ordering'], \
    #                ascending=[True, True, True])
    # b1 = b1.drop(columns=['Ordering'])
    b1.to_csv(config.models_out_dir + '/' + 'all_model_best1.csv', index=False)

    # plot it by forecasting time (simple bars)
    # One figure per forecasting time
    # in order to assign teh same colors I have to do some workaround
    b1['tmp_est'] = b1['Estimator'].map(lambda x: x if x in mlsettings.benchmarks else 'ML')
    # colors = {'Cat1': "#F28E2B", 'Cat2': "#4E79A7", 'Cat3': "#79706E"}
    colors = {'ML': "#0000FF", 'Null_model': "#969696", 'PeakNDVI': "#FF0000", 'Trend': "#009600"}
    for t in b1[var4time].unique():
        crops = b1['Crop'].unique()
        fig, axs = plt.subplots(ncols=len(crops), figsize=(14, 6))
        ax_c = 0 #ax counter
        # get mas metirc
        ymax = b1[b1[var4time] == t][metric2use].max()
        for crop in crops:
            # in order to assign teh same colors I have to do some workaround
            tmp = b1[(b1[var4time] == t) & (b1['Crop'] == crop)]
            sort_dict = {'Null_model': 0, 'Trend': 1, 'PeakNDVI': 2, 'ML': 3}
            tmp['pltOrder'] = tmp['tmp_est'].map(sort_dict)
            tmp = tmp.sort_values('pltOrder')
            p = sns.barplot(tmp, x="tmp_est", y=metric2use, hue="tmp_est",
                       palette=colors, ax=axs[ax_c], dodge=False, width=0.4)
            ml_row = tmp[tmp['tmp_est'] == 'ML']
            [info_string0, info_string1, info_string2, info_string3] = output_row_to_ML_info_string(ml_row, metric2use)
            axs[ax_c].text(0.65, 0.95, info_string0,  transform=axs[ax_c].transAxes, horizontalalignment='center')
            axs[ax_c].text(0.65, 0.91, info_string1, transform=axs[ax_c].transAxes, horizontalalignment='center')
            axs[ax_c].text(0.65, 0.87, info_string2, transform=axs[ax_c].transAxes, horizontalalignment='center')
            axs[ax_c].text(0.65, 0.83, info_string3, transform=axs[ax_c].transAxes, horizontalalignment='center')
            axs[ax_c].get_legend().set_visible(False)
            axs[ax_c].set_title(crop)
            axs[ax_c].set(ylim=(0, ymax*1.1))
            axs[ax_c].set(xlabel='Model')
            ax_c = ax_c + 1

        h, l = p.get_legend_handles_labels()
        #l, h = zip(*sorted(zip(l, h)))
        # p.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)
        plt.legend(h, l, title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        fig.tight_layout()
        plt.savefig(config.models_out_dir + '/' + 'all_model_best1_forecast_time_'+str(t)+'.png')

    # now scatterplot
    crops = b1['Crop'].unique()
    forcTimes = b1[var4time].unique()
    for c in crops:
        for t in forcTimes:
            df_c_t = b1[(b1['Crop'] == c) & (b1[var4time] == t)]
            sort_dict = {'Null_model': 0, 'Trend': 1, 'PeakNDVI': 2, 'ML': 3}
            df_c_t['pltOrder'] = df_c_t['tmp_est'].map(sort_dict)
            df_c_t = df_c_t.sort_values('pltOrder')
            # get the run_ids, put  first
            Ml_run_id = df_c_t[df_c_t['tmp_est'] == 'ML']['runID']
            bench_run_id = df_c_t[df_c_t['tmp_est'].isin(mlsettings.benchmarks)]['runID']
            model_labels = ['ML'] + df_c_t[df_c_t['Estimator'].isin(mlsettings.benchmarks)]['Estimator'].tolist()

            fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)

            for i, ax in enumerate(axs.flatten()):
                # open mres
                print(mres_fns)
                df = pd.read_csv(mres_fns[i])
                lims = [np.floor(np.min([df['yLoo_true'].values, df['yLoo_pred'].values])),
                        np.ceil(np.max([df['yLoo_true'].values, df['yLoo_pred'].values]))]
                r2p = Model_error_stats.r2_nan(df['yLoo_true'].values, df['yLoo_pred'].values)
                for au_code in df['AU_code'].unique():
                    x = df[df['AU_code'] == au_code]['yLoo_true'].values
                    y = df[df['AU_code'] == au_code]['yLoo_pred'].values
                    lbl = df_regNames[df_regNames['AU_code'] == au_code.astype('int')]['AU_name'].values[0]
                    ax.scatter(x, y, label=lbl)
                    ax.plot(lims, lims, color='black', linewidth=0.5)
                    ax.set_title(model_labels[i] + ',R2p=' + str(np.round(r2p, 2)))
                    ax.set_xlim(lims)
                    ax.set_ylim(lims)
                    ax.set_xlabel('Obs')
                    ax.set_ylabel('Pred')
                    ax.legend(frameon=False, loc='upper left')

            fig.savefig(os.path.join(c_dir, c + 'forecst_time_' + str(t) + '.png'))
            plt.close(fig)


    # best configuration of each model type by forecast time, crop type and y var
    # find out, for each of them, the best performing configuration y lead time, crop type and y var
    bmc = mo.groupby(['Crop', var4time, 'Estimator']).apply(lambda x: x.sort_values([metric2use], ascending = sortAscending).head(1)).reset_index(drop=True)
    bmc = bmc.sort_values(['Crop', var4time, metric2use], \
                        ascending=[True, True, sortAscending])
    bmc.to_csv(config.models_out_dir + '/' + 'best_conf_of_all_models.csv', index=False)





    # # make some plots on best 4
    # crop_names = mo['Crop'].unique()
    # # get timing of forecasts
    # # project = b05_Init.init(target)
    # # # get the avg pheno
    # # dirOut = project['output_dir']
    # # pheno_avg = pd.read_csv(dirOut + '/' + project['AOI'] + '_pheno_mean_used.csv')
    # # I have config.sos and config.eos
    #
    #
    # for crop_name in crop_names:
    #     print(crop_name)
    #     fig, axs = plt.subplots(figsize=(12, 3), constrained_layout=True)
    #     time_samp = 'M'
    #
    #     df = b4[b4['Crop'] == crop_name]
    #     # get the forecasting time
    #     xVals = df[var4time].unique()
    #     best_model_OHE = []
    #     best_model_YieldFromTrend = []
    #     best_model_VarSet = []
    #     best_model_names = []
    #     best_model_stat = []
    #     best_model_nonOHE_ft_n = []
    #     best_model_ft_sel = []
    #     best_model_ft_sel_n = []
    #     peak_model_stat = []
    #     null_model_stat = []
    #     trend_model_stat = []
    #     for x in xVals:
    #         df_up2timeID = df[df[var4time] == x]
    #         # get best model name and stat, only looking at ML models (no peak, no null)
    #         dfML_up2timeID = df_up2timeID[df_up2timeID['Estimator'].isin(mlsettings.benchmarks) == False] #isin(mlsettings.benchmarks) == False
    #         # get the single best ML model
    #         # test that there is at least one
    #         if len(dfML_up2timeID[metric2use].values) == 0:
    #             print('no ML model at specific lead time')
    #         if metric2use == 'R2_p':
    #             best_model_row = dfML_up2timeID.iloc[dfML_up2timeID[metric2use].argmax()]
    #         elif metric2use == 'RMSE_p':
    #             best_model_row = dfML_up2timeID.iloc[dfML_up2timeID[metric2use].argmin()]
    #         elif metric2use == 'RMSE_val':
    #             best_model_row = dfML_up2timeID.iloc[dfML_up2timeID[metric2use].argmin()]
    #         elif metric2use == 'rRMSE_p':
    #             best_model_row = dfML_up2timeID.iloc[dfML_up2timeID[metric2use].argmin()]
    #         else:
    #             print('The metric is not coded, the function cannot be executed')
    #             return -1
    #
    #         best_model_names.append(best_model_row['Estimator'])
    #         best_model_stat.append(best_model_row[metric2use])
    #         # get if OHE was used, trend, and the var set
    #         if best_model_row['DoOHEnc'] != 'none':
    #             if best_model_row['DoOHEnc'] == 'AU_level':
    #                 best_model_OHE.append('OHE_au')
    #             elif best_model_row['DoOHEnc'] == 'Cluster_level':
    #                 best_model_OHE.append('OHE_clst')
    #             elif np.isnan(best_model_row['DoOHEnc']):
    #                 best_model_OHE.append('')
    #         else:
    #             best_model_OHE.append('')
    #         if 'AddYieldTrend' in best_model_row:
    #             if best_model_row['AddYieldTrend'] == True:
    #                 best_model_YieldFromTrend.append('YT')
    #             else:
    #                 best_model_YieldFromTrend.append('')
    #         else:
    #             best_model_YieldFromTrend.append('')
    #         # now variables
    #         best_model_nonOHE_ft_n.append(best_model_row['N_features']-best_model_row['N_OHE'])
    #         best_model_ft_sel.append(best_model_row['Ft_selection'])
    #         best_model_ft_sel_n.append(best_model_row['N_selected_fit'])
    #         varsList = ast.literal_eval(best_model_row['Features']) #best_model_row['Features'][1:-1].split(',')
    #         #remove OHE
    #         varsList = [x for x in varsList if not ('OHE' in x or 'YieldFromTrend' in x) ] #[x for x in varsList if not 'OHE' in x]
    #         # remove numbers
    #         remove_digits = str.maketrans('', '', digits)
    #         varsList = [x.translate(remove_digits)[0:-1] for x in varsList] #-2 to remove P or M"
    #         # get unique
    #         varsList = list(set(varsList))
    #         bm_set = 'set not defined'
    #         # check if PCA was activated
    #         PCA_activated = False
    #         if any(['_PC' in x for x in varsList]) == True:
    #             # remove _PC to allow assigning the feature set
    #             varsList = [x.replace('_PC','') for x in varsList]
    #             PCA_activated = True
    #
    #         for key in varSetDict.keys():
    #             if set(varSetDict[key]) == set(varsList):
    #                 bm_set = key
    #         if PCA_activated == True:
    #             bm_set = 'PCA_' + bm_set
    #         # if bm_set == 'set not defined':
    #         #     print('debug PCA')
    #         best_model_VarSet.append(bm_set)
    #
    #         peak_model_row = df_up2timeID[df_up2timeID['Estimator'] == 'PeakNDVI']
    #         peak_model_stat.append(peak_model_row[metric2use].iloc[0])
    #         # null model stats (I have to find it in mo)
    #         null_model_row = mo[(mo['Crop'] == crop_name) & (mo['targetVar'] == y_var) & (mo['Estimator'] == 'Null_model')]
    #         null_model_stat.append(null_model_row[metric2use].iloc[0])
    #         # trend model stats (I have to find it in mo)
    #         trend_model_row = mo[
    #             (mo['Crop'] == crop_name) & (mo['targetVar'] == y_var) & (mo['Estimator'] == 'Trend')]
    #         # in Algeria this was not there
    #         if len(trend_model_row) >0:
    #             trend_model_stat.append(trend_model_row[metric2use].iloc[0])
    #         else:
    #             trend_model_stat.append(np.nan)
    #         print(best_model_names)
    #         print(best_model_stat)
    #         print(peak_model_stat)
    #         print(trend_model_stat)
    #
    #         # plot and save
    #         def progMonth2Month(x):
    #             if x > 12:
    #                 x = x -12
    #             return x
    #
    #
    #         xlabel = 'Time (Month-day)'
    #         up2month = pheno_avg[pheno_avg['Stage'] =='sos']['Month'].to_numpy() + xVals - 1 # minus 1 needed, first vals =1
    #
    #         up2month = list(map(progMonth2Month, up2month))
    #         up2Date = [datetime.datetime(2000, x, 1) for x in up2month]
    #
    #         # reorder years
    #         up2Date_dYear = np.zeros(len(up2Date)).astype(int)
    #         # Find timing that are later than last time, and must go the year before
    #         sub = [i for i, x in enumerate(up2Date) if x > up2Date[-1]]
    #         up2Date_dYear[sub] = -1
    #         up2Date = [datetime.date(x.year + y, x.month, x.day) for x, y in zip(up2Date, up2Date_dYear)]
    #
    #         # x-limits sos to eos
    #         xlim = pheno_avg[pheno_avg['Stage'].isin(['sos', 'eos'])]['Start_stage']
    #
    #         xlim = [datetime.datetime.strptime(x, '%Y-%m-%d') for x in xlim]
    #         # TODO fix by getting day 1 of the sos and last day of EOS and add 15 days each side
    #         xlim[0] = datetime.date(xlim[0].year, xlim[0].month, 1)
    #         #xlim[1] = datetime.date(xlim[1].year, xlim[1].month + 1, 1)
    #         #xlim[1] = xlim[1] + relativedelta(months=+1)
    #         #xlim[1] =np.datetime64(xlim[1] + datetime.timedelta(months=1))
    #         xlim[1] = datetime.date(xlim[1].year, xlim[1].month, 1)
    #         xlim = [np.datetime64(xlim[0] - datetime.timedelta(days=15)),
    #                         np.datetime64(xlim[1] + datetime.timedelta(days=15))]
    #         fmt = mdates.DateFormatter('%b-%d')
    #
    #         if metric2use == 'R2_p':
    #             axs.set_ylim(0.0, 1.0)
    #             yoff4PeakMetric = -15
    #             yoff4ModelName = 25
    #             yoff4ModelNameSecondLine = 10
    #         elif metric2use == 'RMSE_p':
    #             axs.set_ylim(0.2, 0.7)
    #             yoff4PeakMetric = 7.5
    #             yoff4ModelName = -25
    #             yoff4ModelNameSecondLine = -37.5
    #         elif metric2use == 'rRMSE_p':
    #             axs.set_ylim(ylim_rRMSE_p)
    #             yoff4PeakMetric = 5 #7.5
    #             yoff4ModelName = -25
    #             yoff4ModelNameSecondLine = -37.5
    #
    #         # Plot the null
    #         xoff = 0 #20
    #         axs.plot(up2Date, null_model_stat, color='grey', linewidth=1, marker='o', label='Null model')
    #         axs.annotate(round(null_model_stat[-1], 2), (up2Date[-1], null_model_stat[-1]), color='grey',
    #                      ha='center', fontsize=8, fontweight='bold',
    #                      textcoords="offset points", xytext=(xoff, yoff4PeakMetric))
    #
    #         # Plot the trend
    #         if target != 'Algeria':
    #             axs.plot(up2Date, trend_model_stat, color='green', linewidth=1, marker='o', label='Trend model')
    #             axs.annotate(round(trend_model_stat[-1], 2), (up2Date[-1], trend_model_stat[-1]), color='green',
    #                      ha='center', fontsize=8, fontweight='bold',
    #                      textcoords="offset points", xytext=(-xoff, yoff4PeakMetric))
    #
    #         # Plot Peak
    #         axs.plot(up2Date, peak_model_stat, color='red', linewidth=1, marker='o', label='Peak NDVI')
    #         # annotate metric for Peak
    #         npArray = np.around(np.array(peak_model_stat), 2)
    #         if metric2use == 'R2_p':
    #             id_maxs = np.where(npArray == npArray.max())[0].tolist()
    #             lbl = r'$R^2_p$'
    #         elif metric2use == 'RMSE_p':
    #             id_maxs = np.where(npArray == npArray.min())[0].tolist()
    #             lbl = r'$RMSE_p$'
    #         elif metric2use == 'rRMSE_p':
    #             id_maxs = np.where(npArray == npArray.min())[0].tolist()
    #             lbl = r'${\rm rRMSE_p\/(\%)}$'
    #
    #         for i, txt in enumerate(peak_model_stat):
    #             if (i in id_maxs) == False:
    #                 axs.annotate(round(peak_model_stat[i], 2), (up2Date[i], peak_model_stat[i]), color='red',
    #                              ha='center', fontsize=8,
    #                              textcoords="offset points", xytext=(0, yoff4PeakMetric))
    #         for i in id_maxs:
    #             axs.annotate(round(peak_model_stat[i], 2), (up2Date[i], peak_model_stat[i]), color='red',
    #                          ha='center', fontsize=9, fontweight='bold',
    #                          textcoords="offset points", xytext=(0, yoff4PeakMetric))
    #
    #
    #
    #         # Plot the best model config
    #         axs.plot(up2Date, best_model_stat, color='blue', linewidth=1, marker='o', label='ML model')
    #         axs.xaxis.set_major_formatter(fmt)
    #         # ML models print R2, best R2 in bold (R2 or any metric)
    #         npArray  = np.around(np.array(best_model_stat),2)
    #         if metric2use == 'R2_p':
    #             id_maxs = np.where(npArray == npArray.max())[0].tolist()
    #         elif metric2use == 'RMSE_p' or metric2use =='rRMSE_p':
    #             id_maxs = np.where(npArray == npArray.min())[0].tolist()
    #
    #         for i, txt in enumerate(best_model_stat):
    #             if (i in id_maxs) == False:
    #                 axs.annotate(round(best_model_stat[i], 2), (up2Date[i], best_model_stat[i]), color='blue',
    #                             ha='center', fontsize=8,
    #                             textcoords = "offset points", xytext = (0, -15))
    #         # max in bold
    #         for i in id_maxs:
    #             axs.annotate(round(best_model_stat[i],2), (up2Date[i], best_model_stat[i]), color='blue',
    #                             ha='center',  fontsize=9, fontweight='bold',
    #                             textcoords="offset points", xytext=(0, -15))
    #
    #         # annotate model name, OHE, Yield trend
    #         #          varset, ft selection (if any) and (n selected)
    #         for i, txt in enumerate(best_model_names):
    #             anStr = best_model_VarSet[i]
    #             if best_model_OHE[i] != '':
    #                 txt = txt.replace('_', ' ') + ',' + best_model_OHE[i].replace('_', '')
    #             if best_model_YieldFromTrend[i] != '':
    #                 txt = txt + ',' + best_model_YieldFromTrend[i]
    #             if best_model_ft_sel[i] != '':
    #                 #tmpArr = np.array(best_model_ft_sel_n)
    #                 #tmpArr[np.isnan(tmpArr)] = -99
    #                 if best_model_ft_sel[i] == 'none':
    #                     anStr = anStr
    #                 else:
    #                     anStr = anStr + ',' + best_model_ft_sel[i] + \
    #                             str(round(best_model_ft_sel_n[i])) + '/' + str(round(best_model_nonOHE_ft_n[i]))
    #                         #'(' + str(round(best_model_ft_sel_n[i])) + '/' + str(round(best_model_nonOHE_ft_n[i])) + ')'
    #             ftsz = 6 #9
    #             axs.annotate(txt, (up2Date[i], best_model_stat[i]), color='blue', ha='center', textcoords="offset points", xytext=(0,yoff4ModelName), fontsize=ftsz)
    #             axs.annotate(anStr, (up2Date[i], best_model_stat[i]),color='blue',  ha='center', textcoords="offset points", xytext=(0, yoff4ModelNameSecondLine), fontsize=ftsz)
    #
    #         if addCECmodel == True:
    #             from dynamic_masking import CEC_model_plot
    #             # plot cec model
    #             df_cec = CEC_model_plot.CEC_model(target)
    #             df_cec_crop = df_cec[df_cec['CropName'] == crop_name]
    #             axs.plot(df_cec_crop['CEC_date'].values, df_cec_crop['rRMSEp'].values, color='black', linewidth=1,
    #                      marker='o', linestyle='dashed', label='CEC')
    #             for i in range(len(df_cec_crop['CEC_date'].values)):
    #                 axs.annotate(round(df_cec_crop['rRMSEp'].values[i], 2),
    #                              (df_cec_crop['CEC_date'].values[i], df_cec_crop['rRMSEp'].values[i]), color='black',
    #                              ha='center', fontsize=9,  # fontweight='bold',
    #                              textcoords="offset points", xytext=(0, yoff4PeakMetric))
    #
    #         if addCECmodel == False:
    #             axs.set_xlim(xlim[0], xlim[1])
    #         #plt.show()
    #         axs.set_ylabel(lbl)
    #         axs.set_xlabel(xlabel)
    #         #axs.set_title(crop_name + ', ' + y_var, fontsize=12)
    #         axs.set_title(crop_name, fontsize=12)
    #         #axs.set_title(axTilte,  fontsize=12)
    #         #fig.suptitle(crop_name + ', ' + y_var, fontsize=14, fontweight='bold')
    #         if addCECmodel == False:
    #             axs.legend(frameon=False, loc='upper left', ncol = len(axs.lines))
    #         else:
    #             axs.legend(frameon=False, loc='upper right')
    #
    #         # dicVals = list(varSetDict.values())
    #         # x0 = np.datetime64(datetime.datetime.strptime(sosTime[0], '%Y-%m-%d') - datetime.timedelta(days=10))
    #         # axs.text(x0, 0.9, 'Var set legend', fontweight='bold')
    #         # spaceDist = 0.075
    #         # for i, entry in enumerate(dicVals):
    #         #     axs.text(x0, 0.9 - spaceDist - 0.025 - i * spaceDist, entry[1] + ': ' + ', '.join(entry[0]))
    #
    #
    #         #plt.show()
    #         if addCECmodel == True:
    #             strFn = os.path.join(dir, 'best_model_in_time_' + crop_name + '_' + y_var + '_withCEC_time_sampling.png')
    #         else:
    #             strFn = os.path.join(dir, 'best_model_in_time_' + crop_name + '_' + y_var + '_time_sampling.png')
    #
    #         #fig.savefig(strFn.replace(" ", ""))
    #         fig.savefig(strFn)
    #
    #         plt.close(fig)
    #
    #
    # ############################################################################################
    # # make a plot of the percentile rank of peak and NDVI among all tested models
    # for crop_name in crop_names:
    #     for y_var in y_variables:
    #         # fig, axs = plt.subplots(nrows=2, figsize=(12, 6), constrained_layout=True)
    #         fig, axs = plt.subplots(figsize=(12, 3), constrained_layout=True)
    #         time_samp = 'M'
    #         df = mo[(mo['Crop'] == crop_name) & (mo['targetVar'] == y_var) & (mo['Time_sampling'] == time_samp)]
    #         # make sure there are no duplicates in benchmark models
    #         dfML = df[df['Estimator'].isin(cst.benchmarks) == False]
    #         dfBench = df[df['Estimator'].isin(cst.benchmarks) == True]
    #         dfBench = dfBench.drop_duplicates(subset=[var4time, 'Estimator', 'targetVar', 'Crop'])
    #         df = dfML.append(dfBench)
    #         # get the up2timeID
    #         pctPeak = []
    #         pctNull = []
    #         pctTrend = []
    #
    #         nTot = []
    #         xVals = np.sort(df[var4time].unique())
    #         for x in xVals:
    #             df_up2timeID = df[df[var4time] == x]
    #             # rank them
    #             df_up2timeID['Percentile_Rank'] = df_up2timeID[metric2use].rank(ascending=False, pct = True)
    #             nTot.append(df_up2timeID.shape[0])
    #             pctPeak.append(df_up2timeID[df_up2timeID['Estimator']=='PeakNDVI']['Percentile_Rank'].values[0])
    #             pctNull.append(df_up2timeID[df_up2timeID['Estimator'] == 'Null_model']['Percentile_Rank'].values[0])
    #             if includeTrendModel == True:
    #                 pctTrend.append(df_up2timeID[df_up2timeID['Estimator'] == 'Trend']['Percentile_Rank'].values[0])
    #
    #         # plot and save
    #         axs.set_ylim(0.0, 1.0)
    #         axs.plot(up2Date, pctNull, color='grey', linewidth=1, marker='o', label='Null model')
    #         axs.plot(up2Date, pctPeak, color='red', linewidth=1, marker='o', label='Peak NDVI')
    #         if includeTrendModel == True:
    #             axs.plot(up2Date, pctTrend, color='green', linewidth=1, marker='o', label='Trend')
    #         axs.xaxis.set_major_formatter(fmt)
    #         axs.set_xlim(xlim[0], xlim[1])
    #         axs.set_ylabel('Percentile')
    #         axs.set_xlabel(xlabel)
    #         axs.set_title(crop_name, fontsize=12)
    #         axs.legend(frameon=False, loc='upper left')
    #         strFn = os.path.join(dir, 'a_percentile_rank_peak_null_trend' + crop_name + '_' + y_var + '_by_time_sampling.png') #dir + '/' + 'a_percentile_rank_peak_null_trend' + crop_name + '_' + y_var + '_by_time_sampling.png'
    #         #fig.savefig(strFn.replace(" ", ""))
    #         fig.savefig(strFn)
    #         plt.close(fig)
    # ############################################################################################
    # # make a plot of the percentile rank of peak and NDVI among all tested models, now one single graph
    # fig, axs = plt.subplots(figsize=(12, 3), constrained_layout=True)
    # lineStyle = ['-', '--', ':']
    # cc = 0
    # #crop_namesCustomOrder = ['Barley', 'Soft wheat', 'Durum wheat']
    # for crop_name in crop_names:
    #     for y_var in y_variables:
    #         time_samp = 'M'
    #         df = mo[(mo['Crop'] == crop_name) & (mo['targetVar'] == y_var) & (mo['Time_sampling'] == time_samp)]
    #         # get the up2timeID
    #         pctPeak = []
    #         pctNull = []
    #         nTot = []
    #         xVals = np.sort(df[var4time].unique())
    #         for x in xVals:
    #             df_up2timeID = df[df[var4time] == x]
    #             # rank them
    #             df_up2timeID['Percentile_Rank'] = df_up2timeID[metric2use].rank(ascending=False, pct = True)
    #             df_up2timeID['Percentile_Rank'] = df_up2timeID['Percentile_Rank']*100
    #             nTot.append(df_up2timeID.shape[0])
    #             if crop_name == 'Soft wheat' and x == 8:
    #                 print('debug')
    #             pctPeak.append(df_up2timeID[df_up2timeID['Estimator']=='PeakNDVI']['Percentile_Rank'].values[0])
    #             pctNull.append(df_up2timeID[df_up2timeID['Estimator'] == 'Null_model']['Percentile_Rank'].values[0])
    #         axs.plot(up2Date, pctNull, color='grey', linewidth=1, marker='o', linestyle= lineStyle[cc], label='Null model '+ crop_name)
    #         axs.plot(up2Date, pctPeak, color='red', linewidth=1, marker='o', linestyle= lineStyle[cc], label='Peak NDVI ' + crop_name)
    #         cc = cc + 1
    #
    # axs.xaxis.set_major_formatter(fmt)
    # axs.set_xlim(xlim[0], xlim[1])
    #
    # #lbl = r'${\rm Percentile\/of\/rRMSE_p\/ distribution\/(\%)}$'
    # lbl = 'Percentile of\n' + r'${\rm rRMSE_p\/ distribution\/(\%)}$'
    # axs.set_ylabel(lbl)
    # axs.set_xlabel(xlabel)
    # #axs.set_title(crop_name, fontsize=12)
    # axs.legend(frameon=False, loc='lower left')
    # axs.set_ylim(0.0, 100)
    # strFn = os.path.join(dir, 'percentile_rank_peak_null_all_crops_' + y_var + '_by_time_sampling.png') #dir + '/' + 'percentile_rank_peak_null_all_crops_' + y_var + '_by_time_sampling.png'
    # fig.savefig(strFn)
    # #fig.savefig(strFn.replace(" ", ""))
    # plt.close(fig)

    # Now for the best model and benchmark make scatter plot
    # teh refrence data is b1
    print('Compare output ended')