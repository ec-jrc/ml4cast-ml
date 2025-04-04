import pathlib
import sys
import pandas as pd
import os
from A_config import a10_config
from E_viz import e100_eval_figs
from D_modelling import d090_model_wrapper, d140_modelStats
from D_modelling import d140_modelStats


def gather_output(config):
    analysisOutputDir = os.path.join(config.models_out_dir, 'Analysis')
    pathlib.Path(analysisOutputDir).mkdir(parents=True, exist_ok=True)
    run_res = list(sorted(pathlib.Path(config.models_out_dir).glob('ID*_output.csv')))
    print('N files = ' + str(len(run_res)))
    print(
        'Missing files are printed, if any ')  # (no warning issued if they are files that were supposed to be skipped (ft sel asked on 1 var)')
    cc = 0
    missing_counter = 0
    if len(run_res) > 0:
        for file_obj in run_res:
            # print(file_obj, cc)
            cc = cc + 1
            try:
                df = pd.read_csv(file_obj)
            except:
                print('Empty file ' + str(file_obj))
            else:
                try:
                    run_id = int(df['runID'][0])  # .split('_')[1])
                except:
                    print('Error in the file ' + str(file_obj))
                else:
                    # date_id = str(df['runID'][0].split('_')[0])
                    # df_updated = df

                    if file_obj == run_res[0]:
                        # it is the first, save with hdr
                        df.to_csv(analysisOutputDir + '/all_model_output.csv', mode='w', header=True, index=False)
                    else:
                        # it is not first, without hdr
                        df.to_csv(analysisOutputDir + '/all_model_output.csv', mode='a', header=False, index=False)
                        # print if something is missing
                        if run_id > run_id0:
                            if (run_id != run_id0 + 1):
                                for i in range(run_id0 + 1, run_id):
                                    print('Non consececutive runids:' + str(i))
                                    missing_counter = missing_counter + 1
                        # else:
                        #     print('Date changed?', date_id, 'new run id', run_id0)
                    run_id0 = run_id
    else:
        print('There is no a single output file')
    print('End of printing missing files')
    if missing_counter > 0:
        print('Missing outputs are present, if you are running RERUN AFTER TUNING, stop and rerun manager_20_tune')


def compare_fast_outputs(config, n, metric2use='rRMSE_p'):  # RMSE_p' #'R2_p'
    # get some vars from mlSettings class
    mlsettings = a10_config.mlSettings(forecastingMonths=0)
    # includeTrendModel = True   #for Algeria run ther is no trend model
    # addCECmodel = False #only for South Africa

    analysisOutputDir = os.path.join(config.models_out_dir, 'Analysis')
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

    mo = pd.read_csv(analysisOutputDir + '/' + 'all_model_output.csv')
    # get best n ML configurations by lead time, crop type and y var PLUS benchmarks
    moML = mo[mo['Estimator'].isin(mlsettings.benchmarks) == False]
    bn = moML.groupby(['Crop', var4time]).apply(
        lambda x: x.sort_values([metric2use], ascending=sortAscending).head(n)).reset_index(drop=True)
    # always add the benchmarks

    bn = bn.sort_values([var4time, 'Crop', metric2use], \
                        ascending=[True, True, sortAscending])
    bn.to_csv(analysisOutputDir + '/' + 'ML_models_to_run_with_tuning.csv', index=False)
    return bn['runID'].tolist()


def compare_outputs(config, fn_shape_gaul1, country_name_in_shp_file, gdf_gaul0_column='name0',
                    metric2use='rRMSE_p'):  # RMSE_p' #'R2_p'
    # get some vars from mlSettings class
    mlsettings = a10_config.mlSettings(forecastingMonths=0)
    # includeTrendModel = True   #for Algeria run ther is no trend model
    # addCECmodel = False #only for South Africa

    analysisOutputDir = os.path.join(config.models_out_dir, 'Analysis')
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

    mo = pd.read_csv(analysisOutputDir + '/' + 'all_model_output.csv')

    # Until 2025 02 18, RMSE_p and rRMSE_p where computed using d140_modelStats.allStats_spatial
    # that take the time average of the RMSE by year with the result that if a year has only few data and
    # it is very good or bad, it will have a lot of weight. We decided then to go always for an overall rmse
    # Here I recompute it; it will not be needed in the new runs
    if not('RMSE_p_spatial' in mo.columns):
        # it is not there, so the current 'RMSE_p', 'rRMSE_p' are spatial
        # I have to rename them, and compute the overall
        mo = mo.rename(columns={'RMSE_p': 'RMSE_p_spatial', 'rRMSE_p': 'rRMSE_p_spatial'})
        mo['RMSE_p'] = None
        mo['rRMSE_p'] = None
        for index, row in mo.iterrows():
            runID = row['runID']            # get run_id
            myID = f'{runID:06d}'
            fn_mRes_out = os.path.join(config.models_out_dir, 'ID_' + str(myID) + '_crop_' + row['Crop'] + '_Yield_' + row['Estimator'] + '_mres.csv')
            if os.path.exists(fn_mRes_out):
                mRes = pd.read_csv(fn_mRes_out)
            else:
                fn_spec = os.path.join(config.models_spec_dir, str(myID) + '_' + row['Crop'] + '_' + row['Estimator'] + '.json')
                mRes = d090_model_wrapper.fit_and_validate_single_model(fn_spec, config, 'tuning', run2get_mres_only=True)
            res = d140_modelStats.allStats_overall(mRes)
            mo.at[index, 'RMSE_p'] = res['Pred_RMSE']
            mo.at[index, 'rRMSE_p'] = res['rel_Pred_RMSE']



    # add the calendar month at which thh forcast can be done
    di = dict(zip(config.forecastingMonths, config.forecastingCalendarMonths))
    mo['forecast_issue_calendar_month'] = mo['forecast_time'].replace(di)
    mo['forecast_issue_calendar_month'] = mo['forecast_issue_calendar_month'].astype(
        int) + 1  # add one month as forecast are made beginiing of the month after
    mo.loc[mo['forecast_issue_calendar_month'] > 12, ['forecast_issue_calendar_month']] = mo[
                                                                                              'forecast_issue_calendar_month'] - 12
    # get best 4 ML configurations by lead time, crop type and y var PLUS benchmarks
    moML = mo[mo['Estimator'].isin(mlsettings.benchmarks) == False]
    b4 = moML.groupby(['Crop', var4time]).apply(
        lambda x: x.sort_values([metric2use], ascending=sortAscending).head(4)).reset_index(drop=True)
    # always add the benchmarks
    tmp = mo.groupby(['Crop', var4time]) \
        .apply(lambda x: x.loc[x['Estimator'].isin(mlsettings.benchmarks)]).reset_index(drop=True)
    tmp = tmp.drop_duplicates(subset=[var4time, 'Estimator', 'Crop'])
    b4 = pd.concat([b4, tmp])
    b4 = b4.sort_values([var4time, 'Crop', metric2use], \
                        ascending=[True, True, sortAscending])
    b4.to_csv(analysisOutputDir + '/' + 'all_model_best4.csv', index=False)

    # and absolute ML best (plus benchmarks)
    # get best 4 ML configurations by lead time, crop type and y var PLUS benchmarks
    moML = mo[mo['Estimator'].isin(mlsettings.benchmarks) == False]
    b1ML = moML.groupby(['Crop', var4time]).apply(
        lambda x: x.sort_values([metric2use], ascending=sortAscending).head(1)).reset_index(drop=True)
    # always add the benchmarks
    tmp = mo.groupby(['Crop', var4time]) \
        .apply(lambda x: x.loc[x['Estimator'].isin(mlsettings.benchmarks)]).reset_index(drop=True)
    tmp = tmp.drop_duplicates(subset=[var4time, 'Estimator', 'Crop'])
    b1 = pd.concat([b1ML, tmp])
    b1.to_csv(analysisOutputDir + '/' + 'all_model_best1.csv', index=False)
    # compute error at AU level
    b1withAUerror = e100_eval_figs.AU_error(b1, config, analysisOutputDir)
    # plot scatter of Ml and bechmark, one plot per crop and forecasting time (this function is saving mRes)
    #e100_eval_figs.scatter_plots_and_maps(b1, config, mlsettings, var4time, analysisOutputDir, fn_shape_gaul1,
    e100_eval_figs.scatter_plots_and_maps(b1withAUerror, config, mlsettings, var4time, analysisOutputDir, fn_shape_gaul1,
                                          country_name_in_shp_file, gdf_gaul0_column=gdf_gaul0_column)
    # plot it by forecasting time (simple bars for each forecasting time), mRes is created above
    e100_eval_figs.bars_by_forecast_time2(b1withAUerror, config, 'rRMSE_p', mlsettings, var4time, analysisOutputDir)
    # e100_eval_figs.bars_by_forecast_time(b1, config, metric2use, mlsettings, var4time, analysisOutputDir)

    print('Compare output ended')

    # best configuration of each model type by forecast time, crop type and y var
    # find out, for each of them, the best performing configuration y lead time, crop type and y var
    # bmc = mo.groupby(['Crop', var4time, 'Estimator']).apply(lambda x: x.sort_values([metric2use], ascending = sortAscending).head(1)).reset_index(drop=True)
    # bmc = bmc.sort_values(['Crop', var4time, metric2use], \
    #                     ascending=[True, True, sortAscending])
    # bmc.to_csv(analysisOutputDir + '/' + 'best_conf_of_all_models.csv', index=False)

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
