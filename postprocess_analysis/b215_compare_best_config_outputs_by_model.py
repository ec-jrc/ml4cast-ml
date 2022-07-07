import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import b05_Init
import datetime
from string import digits
import ast
import src.constants as cst
import os
from dateutil.relativedelta import *
import pathlib

def compare_outputs (dir, target):
    includeTrendModel = True   #for Algeria run ther is no trend model
    ylim_rRMSE_p = [0,30]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 400)  # width is = 400
    metric2use = 'rRMSE_p' #RMSE_p' #'R2_p'
    if metric2use == 'R2_p':
        sortAscending = False
    elif metric2use == 'RMSE_p':
        sortAscending = True
    elif metric2use == 'rRMSE_p':
        sortAscending = True
    else:
        print('The metric is not coded, the function cannot be executed')
        return -1

    if target == 'Algeria':
        #correct issue of calling forecast_time as lead_time, fixed after the Algeria paper
        var4time = 'lead_time'
    else:
        var4time ='forecast_time'

    mo = pd.read_csv(dir + '/' + 'all_model_output.csv')

    # best 4 ML configurations by lead time, crop type and y var PLUS benchmarks
    #'Now forecast_time, for algeria/paper must be lead_time'
    moML = mo[mo['Estimator'].isin(cst.benchmarks) == False]
    b4 = moML.groupby(['Crop','targetVar','Time_sampling',var4time]).apply(lambda x: x.sort_values([metric2use], ascending = sortAscending).head(4)).reset_index(drop=True)
    # always add the benchmarks

    tmp = mo.groupby(['Crop','targetVar','Time_sampling',var4time]) \
        .apply(lambda x: x.loc[x['Estimator'].isin(cst.benchmarks)]).reset_index(drop=True)
    tmp = tmp.drop_duplicates(subset=[var4time, 'Estimator', 'targetVar', 'Crop'])
    b4 = b4.append(tmp)
    #b4.to_csv(dirOutModel + '/' + project['AOI'] + '_model_best4.csv')
    b4 = b4.sort_values(['Time_sampling',var4time,'Crop','targetVar',metric2use], \
                   ascending=[True,True,True,True,sortAscending])
    b4.to_csv(dir + '/' + 'all_model_best4.csv', index=False)

    # and absolute best
    b1 = mo.groupby(['Crop', 'targetVar', 'Time_sampling', var4time]).apply(
        lambda x: x.sort_values([metric2use], ascending=sortAscending).head(1)).reset_index(drop=True)
    b1['Ordering'] = 0  # just used to orde Ml, pean, null in the csv
    for idx, val in enumerate(cst.benchmarks, start=0):
        tmp = mo.groupby(['Crop', 'targetVar', 'Time_sampling', var4time]) \
            .apply(lambda x: x.loc[x['Estimator'].isin([cst.benchmarks[idx]])]).reset_index(drop=True)
        tmp['Ordering'] = idx+1
        tmp = tmp.drop_duplicates(subset=[var4time, 'Estimator', 'targetVar', 'Crop'])
        b1 = b1.append(tmp)

    # add feature set name, attribute the column "Features" to a var set
    # remove OHE vars
    b1['feat_set'] = b1['Features'].map(lambda x: [y for y in ast.literal_eval(x) if not 'OHE' in y])
    # remove YieldFromTrend vars
    b1['feat_set'] = b1['feat_set'].map(lambda x: [y for y in x if not 'Trend' in y])
    # remove digits
    remove_digits = str.maketrans('', '', digits)
    b1['feat_set'] = b1['feat_set'].map(lambda x: [y.translate(remove_digits)[0:-1] for y in x])
    # get uniques
    b1['feat_set'] = b1['feat_set'].map(lambda x: list(set(x)))
    # give name to var set
    # feature_groups = {
    #     'rs_met': ['ND', 'NDmax', 'Rad', 'RainSum', 'T', 'Tmin', 'Tmax'],
    #     'rs_met_reduced': ['ND', 'RainSum', 'T'],
    #     'rs_met_sm_reduced': ['ND', 'RainSum', 'T', 'SM'],  # test of ZA
    #     'rs': ['ND', 'NDmax'],
    #     'rs_reduced': ['ND'],
    #     'rs_sm_reduced': ['ND', 'SM'],
    #     'met': ['Rad', 'RainSum', 'T', 'Tmin', 'Tmax'],
    #     'met_reduced': ['Rad', 'RainSum', 'T'],
    #     'met_sm_reduced': ['Rad', 'RainSum', 'T', 'SM']
    # }
    # # dictionary for group labels used in plots
    # feature_groups2labels = {
    #     'rs_met':  'RS&Met',
    #     'rs_met_reduced': 'RS&Met-',
    #     'rs_met_sm_reduced': 'SM&RS&Met-',
    #     'rs': 'RS',
    #     'rs_reduced': 'RS-',
    #     'rs_sm_reduced': 'SM&RS-',
    #     'met': 'Met',
    #     'met_reduced': 'Met-',
    #     'met_sm_reduced': 'SM&Met-'
    # }

    varSetDict = cst.feature_groups
    old_keys = list(varSetDict.keys())
    for key in old_keys:
        # print(key)
        # print(cst.feature_groups2labels[key])
        varSetDict[cst.feature_groups2labels[key]] = varSetDict.pop(key)

    # varSetDict = {
    #     'RS&Met': ['ND', 'NDmax', 'Rad', 'RainSum', 'T', 'Tmin', 'Tmax'],
    #     'RS&Met-': ['ND', 'RainSum', 'T'],
    #     'RS': ['ND', 'NDmax'],
    #     'RS-': ['ND'],
    #     'Met': ['Rad', 'RainSum', 'T', 'Tmin', 'Tmax'],
    #     'Met-': ['Rad', 'RainSum', 'T'],
    #     'peak': ['NDpeak'],
    #     'null': ['Non']
    # }
    for key in varSetDict.keys():
        b1['feat_set'] = b1['feat_set'].map(lambda x: key if set(x) == set(varSetDict[key]) else x)
    b1['feat_set'] = b1['feat_set'].map(lambda x: 'n.a.' if set(x) == set(['NDpea']) else x)
    b1['feat_set'] = b1['feat_set'].map(lambda x: 'n.a.' if set(x) == set(['Non']) else x)

    # b1['feat_set'] = b1['feat_set'].map(lambda x: 'RS&Met' if set(x) == set(varSetDict['RS&Met']) else x)
    # b1['feat_set'] = b1['feat_set'].map(lambda x: 'RS&Met-' if set(x) == set(varSetDict['RS&Met-']) else x)
    # b1['feat_set'] = b1['feat_set'].map(lambda x: 'RS' if set(x) == set(varSetDict['RS']) else x)
    # b1['feat_set'] = b1['feat_set'].map(lambda x: 'RS-' if set(x) == set(varSetDict['RS-']) else x)
    # b1['feat_set'] = b1['feat_set'].map(lambda x: 'Met' if set(x) == set(varSetDict['Met']) else x)
    # b1['feat_set'] = b1['feat_set'].map(lambda x: 'Met-' if set(x) == set(varSetDict['Met-']) else x)
    # b1['feat_set'] = b1['feat_set'].map(lambda x: 'n.a.' if set(x) == set(varSetDict['peak']) else x)
    # b1['feat_set'] = b1['feat_set'].map(lambda x: 'n.a.' if set(x) == set(varSetDict['null']) else x)
    # custom_dict = {'Barley': 0, 'Soft wheat': 1, 'Durum wheat': 3}
    # b1['tmp'] = b1['Crop'].map(custom_dict)
    b1 = b1.sort_values(['Crop','Time_sampling',var4time,'targetVar','Ordering'], \
                   ascending=[True,True,True,True,True])
    b1 = b1.drop(columns=['Ordering'])
    b1.to_csv(dir + '/' + 'all_model_best1.csv', index=False)


    # best configuration of each model by forecast time, crop type and y var
    # find out the models tested
    # models = mo['Estimator'].unique()
    # find out, for each of them, the best performing configuration y lead time, crop type and y var
    bmc = mo.groupby(['Crop','targetVar','Time_sampling',var4time,'Estimator']).apply(lambda x: x.sort_values([metric2use], ascending = sortAscending).head(1)).reset_index(drop=True)
    bmc = bmc.sort_values(['Time_sampling', 'Crop', 'targetVar', var4time, metric2use], \
                        ascending=[True, True, True, True, sortAscending])
    bmc.to_csv(dir + '/' + 'best_conf_of_all_models.csv', index=False)

    # average performance (in all crops, by target variable, by model specs)
    # first remove OHE from features as we get a different list by crops
    #mo = mo.fillna('')
    mo['Features_withoutOHE'] = mo['Features'].apply(lambda x: '['+', '.join(str(item) for item in str(x).strip('][').split(', ') if not("OHE" in item)) +']')
    avgPerf = mo.groupby(['dataScaling','DoOHEnc','AddTargetMeanToFeature',
                          #'DoScaleOHEnc','Features_withoutOHE','scoringMetric',
                          'Features_withoutOHE', 'scoringMetric',
                          'Time_sampling', var4time,
                          'targetVar','Estimator'], dropna=False).mean()
    #avgPerf.to_csv(dirOutModel + '/' + project['AOI'] + '_model_mean_perf.csv')
    avgPerf.to_csv(dir + '/' + 'all_model_mean_perf.csv') #, index=False) sort_values(['Crop','targetVar','Time_sampling','up2timeID'], ascending=True)\


    # make some plots on best 4
    stat_column_name = metric2use       # select what to plot
    #crop_names = ['Barley','Durum wheat','Soft wheat']  # temporary, it should be read from file
    crop_names = mo['Crop'].unique()
    # get timing of forecasts
    project = b05_Init.init(target)
    # get the avg pheno
    dirOut = project['output_dir']
    pheno_avg = pd.read_csv(dirOut + '/' + project['AOI'] + '_pheno_mean_used.csv')
    # y variables to be predicted
    # y_variables= ['Yield', 'Production']
    y_variables = mo['targetVar'].unique() #['Yield']
#    time_samplings = mo['Time_sampling'].unique() #['M'] #['M','P']
    for crop_name in crop_names:
        print(crop_name)
        for y_var in y_variables:
            #fig, axs = plt.subplots(nrows=2, figsize=(12, 6), constrained_layout=True)
            fig, axs = plt.subplots(figsize=(12, 3), constrained_layout=True)
            time_samp = 'M'

            df = b4[(b4['Crop'] == crop_name) & (b4['targetVar'] == y_var) & (b4['Time_sampling'] == time_samp)]
            # get the up2timeID
            xVals = df[var4time].unique()
            best_model_OHE = []
            best_model_YieldFromTrend = []
            best_model_VarSet = []
            best_model_names = []
            best_model_stat = []
            best_model_nonOHE_ft_n = []
            best_model_ft_sel = []
            best_model_ft_sel_n = []
            peak_model_stat = []
            null_model_stat = []
            trend_model_stat = []
            for x in xVals:
                df_up2timeID = df[df[var4time] == x]
                # get best model name and stat, only looking at ML models (no peak, no null)
                dfML_up2timeID = df_up2timeID[df_up2timeID['Estimator'].isin(cst.benchmarks) == False]
                # get the single best ML model
                # test that there is at least one
                if len(dfML_up2timeID[stat_column_name].values) == 0:
                    print('no ML model at specific lead time')
                if metric2use == 'R2_p':
                    best_model_row = dfML_up2timeID.iloc[dfML_up2timeID[stat_column_name].argmax()]
                elif metric2use == 'RMSE_p':
                    best_model_row = dfML_up2timeID.iloc[dfML_up2timeID[stat_column_name].argmin()]
                elif metric2use == 'rRMSE_p':
                    best_model_row = dfML_up2timeID.iloc[dfML_up2timeID[stat_column_name].argmin()]
                else:
                    print('The metric is not coded, the function cannot be executed')
                    return -1

                # get the id and load the mRer file (y pre and y act)
                # runID = best_model_row['runID']
                # #mRes_fn = list(pathlib.Path(dir).glob(f'*{runID}*_mRes.csv'))
                best_model_names.append(best_model_row['Estimator'])
                best_model_stat.append(best_model_row[stat_column_name])
                # get if OHE was used, trend, and the var set
                if best_model_row['DoOHEnc'] != 'none':
                    if best_model_row['DoOHEnc'] == 'AU_level':
                        best_model_OHE.append('OHE_au')
                    elif best_model_row['DoOHEnc'] == 'Cluster_level':
                        best_model_OHE.append('OHE_clst')
                    elif np.isnan(best_model_row['DoOHEnc']):
                        best_model_OHE.append('')
                else:
                    best_model_OHE.append('')
                if 'AddYieldTrend' in best_model_row:
                    if best_model_row['AddYieldTrend'] == True:
                        best_model_YieldFromTrend.append('YT')
                    else:
                        best_model_YieldFromTrend.append('')
                else:
                    best_model_YieldFromTrend.append('')
                # now variables
                best_model_nonOHE_ft_n.append(best_model_row['N_features']-best_model_row['N_OHE'])
                best_model_ft_sel.append(best_model_row['Ft_selection'])
                best_model_ft_sel_n.append(best_model_row['N_selected_fit'])
                varsList = ast.literal_eval(best_model_row['Features']) #best_model_row['Features'][1:-1].split(',')
                #remove OHE
                varsList = [x for x in varsList if not ('OHE' in x or 'YieldFromTrend' in x) ] #[x for x in varsList if not 'OHE' in x]
                # remove numbers
                remove_digits = str.maketrans('', '', digits)
                varsList = [x.translate(remove_digits)[0:-1] for x in varsList] #-2 to remove P or M"
                # get unique
                varsList = list(set(varsList))
                bm_set = 'set not defined'
                if crop_name == 'Soybeans' and x == 6:
                    print()
                for key in varSetDict.keys():
                    if set(varSetDict[key]) == set(varsList):
                        bm_set = key
                best_model_VarSet.append(bm_set)
                # varSetDict = {
                #     'full': [['ND', 'NDmax', 'Rad', 'RainSum', 'T', 'Tmin', 'Tmax'], 'RS&Met'],
                #     'full_reduced': [['ND', 'RainSum', 'T'], 'RS&Met-'],
                #     'rs':   [['ND', 'NDmax'], 'RS'],
                #     'rs_reduced': [['ND'], 'RS-'],
                #     'met': [['Rad', 'RainSum', 'T', 'Tmin', 'Tmax'], 'Met'],
                #     'met_reduced': [['Rad', 'RainSum', 'T'], 'Met-'],
                #     'full_reduced': [['ND', 'RainSum', 'T'], 'RS&Met-']
                # }
                # if set(varSetDict['full'][0]) == set(varsList):
                #     best_model_VarSet.append(varSetDict['full'][1])
                # elif set(varSetDict['rs'][0]) == set(varsList):
                #     best_model_VarSet.append(varSetDict['rs'][1])
                # elif set(varSetDict['rs_reduced'][0]) == set(varsList):
                #     best_model_VarSet.append(varSetDict['rs_reduced'][1])
                # elif set(varSetDict['met'][0]) == set(varsList):
                #     best_model_VarSet.append(varSetDict['met'][1])
                # elif set(varSetDict['met_reduced'][0]) == set(varsList):
                #     best_model_VarSet.append(varSetDict['met_reduced'][1])
                # elif set(varSetDict['full_reduced'][0]) == set(varsList):
                #     best_model_VarSet.append(varSetDict['full_reduced'][1])
                # else:
                #     best_model_VarSet.append('set not defined')#: ' + ', '.join(varsList))
                # peak_model stats
                peak_model_row = df_up2timeID[df_up2timeID['Estimator'] == 'PeakNDVI']
                peak_model_stat.append(peak_model_row[stat_column_name].iloc[0])
                # null model stats (I have to find it in mo)
                null_model_row = mo[(mo['Crop'] == crop_name) & (mo['targetVar'] == y_var) & (mo['Estimator'] == 'Null_model')]
                null_model_stat.append(null_model_row[stat_column_name].iloc[0])
                # trend model stats (I have to find it in mo)
                trend_model_row = mo[
                    (mo['Crop'] == crop_name) & (mo['targetVar'] == y_var) & (mo['Estimator'] == 'Trend')]
                # in Algeria this was not there
                if len(trend_model_row) >0:
                    trend_model_stat.append(trend_model_row[stat_column_name].iloc[0])
                else:
                    trend_model_stat.append(np.nan)
            print(best_model_names)
            print(best_model_stat)
            print(peak_model_stat)
            print(trend_model_stat)

            # plot and save
            def progMonth2Month(x):
                if x > 12:
                    x = x -12
                return x


            xlabel = 'Time (Month-day)'
            up2month = pheno_avg[pheno_avg['Stage'] =='sos']['Month'].to_numpy() + xVals - 1 # minus 1 needed, first vals =1

            up2month = list(map(progMonth2Month, up2month))
            up2Date = [datetime.datetime(2000, x, 1) for x in up2month]

            # reorder years
            up2Date_dYear = np.zeros(len(up2Date)).astype(int)
            # Find timing that are later than last time, and must go the year before
            sub = [i for i, x in enumerate(up2Date) if x > up2Date[-1]]
            up2Date_dYear[sub] = -1
            up2Date = [datetime.date(x.year + y, x.month, x.day) for x, y in zip(up2Date, up2Date_dYear)]

            # x-limits sos to eos
            xlim = pheno_avg[pheno_avg['Stage'].isin(['sos', 'eos'])]['Start_stage']

            xlim = [datetime.datetime.strptime(x, '%Y-%m-%d') for x in xlim]
            # TODO fix by getting day 1 of the sos and last day of EOS and add 15 days each side
            xlim[0] = datetime.date(xlim[0].year, xlim[0].month, 1)
            #xlim[1] = datetime.date(xlim[1].year, xlim[1].month + 1, 1)
            #xlim[1] = xlim[1] + relativedelta(months=+1)
            #xlim[1] =np.datetime64(xlim[1] + datetime.timedelta(months=1))
            xlim[1] = datetime.date(xlim[1].year, xlim[1].month, 1)
            xlim = [np.datetime64(xlim[0] - datetime.timedelta(days=15)),
                            np.datetime64(xlim[1] + datetime.timedelta(days=15))]
            fmt = mdates.DateFormatter('%b-%d')

            if metric2use == 'R2_p':
                axs.set_ylim(0.0, 1.0)
                yoff4PeakMetric = -15
                yoff4ModelName = 25
                yoff4ModelNameSecondLine = 10
            elif metric2use == 'RMSE_p':
                axs.set_ylim(0.2, 0.7)
                yoff4PeakMetric = 7.5
                yoff4ModelName = -25
                yoff4ModelNameSecondLine = -37.5
            elif metric2use == 'rRMSE_p':
                axs.set_ylim(ylim_rRMSE_p)
                yoff4PeakMetric = 7.5
                yoff4ModelName = -25
                yoff4ModelNameSecondLine = -37.5

            # Plot the null
            xoff = 20
            axs.plot(up2Date, null_model_stat, color='grey', linewidth=1, marker='o', label='Null model')
            axs.annotate(round(null_model_stat[-1], 2), (up2Date[-1], null_model_stat[-1]), color='grey',
                         ha='center', fontsize=8, fontweight='bold',
                         textcoords="offset points", xytext=(xoff, yoff4PeakMetric))
            # Plot the trend
            if target != 'Algeria':
                axs.plot(up2Date, trend_model_stat, color='green', linewidth=1, marker='o', label='Trend model')
                axs.annotate(round(trend_model_stat[-1], 2), (up2Date[-1], trend_model_stat[-1]), color='green',
                         ha='center', fontsize=8, fontweight='bold',
                         textcoords="offset points", xytext=(-xoff, yoff4PeakMetric))

            # Plot Peak
            axs.plot(up2Date, peak_model_stat, color='red', linewidth=1, marker='o', label='Peak NDVI')
            # annotate metric for Peak
            npArray = np.around(np.array(peak_model_stat), 2)
            if metric2use == 'R2_p':
                id_maxs = np.where(npArray == npArray.max())[0].tolist()
                lbl = r'$R^2_p$'
            elif metric2use == 'RMSE_p':
                id_maxs = np.where(npArray == npArray.min())[0].tolist()
                lbl = r'$RMSE_p$'
            elif metric2use == 'rRMSE_p':
                id_maxs = np.where(npArray == npArray.min())[0].tolist()
                lbl = r'${\rm rRMSE_p\/(\%)}$'

            for i, txt in enumerate(peak_model_stat):
                if (i in id_maxs) == False:
                    axs.annotate(round(peak_model_stat[i], 2), (up2Date[i], peak_model_stat[i]), color='red',
                                 ha='center', fontsize=8,
                                 textcoords="offset points", xytext=(0, yoff4PeakMetric))
            for i in id_maxs:
                axs.annotate(round(peak_model_stat[i], 2), (up2Date[i], peak_model_stat[i]), color='red',
                             ha='center', fontsize=9, fontweight='bold',
                             textcoords="offset points", xytext=(0, yoff4PeakMetric))

            # Plot the best model config
            axs.plot(up2Date, best_model_stat, color='blue', linewidth=1, marker='o', label='ML model')
            axs.xaxis.set_major_formatter(fmt)
            # ML models print R2, best R2 in bold (R2 or any metric)
            npArray  = np.around(np.array(best_model_stat),2)
            if metric2use == 'R2_p':
                id_maxs = np.where(npArray == npArray.max())[0].tolist()
            elif metric2use == 'RMSE_p' or metric2use =='rRMSE_p':
                id_maxs = np.where(npArray == npArray.min())[0].tolist()

            for i, txt in enumerate(best_model_stat):
                if (i in id_maxs) == False:
                    axs.annotate(round(best_model_stat[i], 2), (up2Date[i], best_model_stat[i]), color='blue',
                                ha='center', fontsize=8,
                                textcoords = "offset points", xytext = (0, -15))
            # max in bold
            for i in id_maxs:
                axs.annotate(round(best_model_stat[i],2), (up2Date[i], best_model_stat[i]), color='blue',
                                ha='center',  fontsize=9, fontweight='bold',
                                textcoords="offset points", xytext=(0, -15))

            # annotate model name, OHE, Yield trend
            #          varset, ft selection (if any) and (n selected)
            for i, txt in enumerate(best_model_names):
                anStr = best_model_VarSet[i]
                if best_model_OHE[i] != '':
                    txt = txt.replace('_', ' ') + ',' + best_model_OHE[i].replace('_', '')
                if best_model_YieldFromTrend[i] != '':
                    txt = txt + ',' + best_model_YieldFromTrend[i]
                if best_model_ft_sel[i] != '':
                    #tmpArr = np.array(best_model_ft_sel_n)
                    #tmpArr[np.isnan(tmpArr)] = -99
                    if best_model_ft_sel[i] == 'none':
                        anStr = anStr
                    else:
                        anStr = anStr + ',' + best_model_ft_sel[i] + \
                                str(round(best_model_ft_sel_n[i])) + '/' + str(round(best_model_nonOHE_ft_n[i]))
                            #'(' + str(round(best_model_ft_sel_n[i])) + '/' + str(round(best_model_nonOHE_ft_n[i])) + ')'
                axs.annotate(txt, (up2Date[i], best_model_stat[i]), color='blue', ha='center', textcoords="offset points", xytext=(0,yoff4ModelName), fontsize=9)
                axs.annotate(anStr, (up2Date[i], best_model_stat[i]),color='blue',  ha='center', textcoords="offset points", xytext=(0, yoff4ModelNameSecondLine), fontsize=9)
            #plt.show()
            axs.set_xlim(xlim[0], xlim[1])
            axs.set_ylabel(lbl)
            axs.set_xlabel(xlabel)
            #axs.set_title(crop_name + ', ' + y_var, fontsize=12)
            axs.set_title(crop_name, fontsize=12)
            #axs.set_title(axTilte,  fontsize=12)
            #fig.suptitle(crop_name + ', ' + y_var, fontsize=14, fontweight='bold')
            axs.legend(frameon=False, loc='upper left', ncol = len(axs.lines))


            # dicVals = list(varSetDict.values())
            # x0 = np.datetime64(datetime.datetime.strptime(sosTime[0], '%Y-%m-%d') - datetime.timedelta(days=10))
            # axs.text(x0, 0.9, 'Var set legend', fontweight='bold')
            # spaceDist = 0.075
            # for i, entry in enumerate(dicVals):
            #     axs.text(x0, 0.9 - spaceDist - 0.025 - i * spaceDist, entry[1] + ': ' + ', '.join(entry[0]))


            #plt.show()
            strFn = os.path.join(dir, 'best_model_in_time_' + crop_name + '_' + y_var + '_time_sampling.png') #dir + '/' + 'best_model_in_time_' + crop_name + '_' + y_var + '_time_sampling.png'
            #fig.savefig(strFn.replace(" ", ""))
            fig.savefig(strFn)

            plt.close(fig)


    ############################################################################################
    # make a plot of the percentile rank of peak and NDVI among all tested models
    for crop_name in crop_names:
        for y_var in y_variables:
            # fig, axs = plt.subplots(nrows=2, figsize=(12, 6), constrained_layout=True)
            fig, axs = plt.subplots(figsize=(12, 3), constrained_layout=True)
            time_samp = 'M'
            df = mo[(mo['Crop'] == crop_name) & (mo['targetVar'] == y_var) & (mo['Time_sampling'] == time_samp)]
            # make sure there are no duplicates in benchmark models
            dfML = df[df['Estimator'].isin(cst.benchmarks) == False]
            dfBench = df[df['Estimator'].isin(cst.benchmarks) == True]
            dfBench = dfBench.drop_duplicates(subset=[var4time, 'Estimator', 'targetVar', 'Crop'])
            df = dfML.append(dfBench)
            # get the up2timeID
            pctPeak = []
            pctNull = []
            pctTrend = []

            nTot = []
            xVals = np.sort(df[var4time].unique())
            for x in xVals:
                df_up2timeID = df[df[var4time] == x]
                # rank them
                df_up2timeID['Percentile_Rank'] = df_up2timeID[metric2use].rank(ascending=False, pct = True)
                nTot.append(df_up2timeID.shape[0])
                pctPeak.append(df_up2timeID[df_up2timeID['Estimator']=='PeakNDVI']['Percentile_Rank'].values[0])
                pctNull.append(df_up2timeID[df_up2timeID['Estimator'] == 'Null_model']['Percentile_Rank'].values[0])
                if includeTrendModel == True:
                    pctTrend.append(df_up2timeID[df_up2timeID['Estimator'] == 'Trend']['Percentile_Rank'].values[0])

            # plot and save
            axs.set_ylim(0.0, 1.0)
            axs.plot(up2Date, pctNull, color='grey', linewidth=1, marker='o', label='Null model')
            axs.plot(up2Date, pctPeak, color='red', linewidth=1, marker='o', label='Peak NDVI')
            if includeTrendModel == True:
                axs.plot(up2Date, pctTrend, color='green', linewidth=1, marker='o', label='Trend')
            axs.xaxis.set_major_formatter(fmt)
            axs.set_xlim(xlim[0], xlim[1])
            axs.set_ylabel('Percentile')
            axs.set_xlabel(xlabel)
            axs.set_title(crop_name, fontsize=12)
            axs.legend(frameon=False, loc='upper left')
            strFn = os.path.join(dir, 'a_percentile_rank_peak_null_trend' + crop_name + '_' + y_var + '_by_time_sampling.png') #dir + '/' + 'a_percentile_rank_peak_null_trend' + crop_name + '_' + y_var + '_by_time_sampling.png'
            #fig.savefig(strFn.replace(" ", ""))
            fig.savefig(strFn)
            plt.close(fig)
    ############################################################################################
    # make a plot of the percentile rank of peak and NDVI among all tested models, now one single graph
    fig, axs = plt.subplots(figsize=(12, 3), constrained_layout=True)
    lineStyle = ['-', '--', ':']
    cc = 0
    #crop_namesCustomOrder = ['Barley', 'Soft wheat', 'Durum wheat']
    for crop_name in crop_names:
        for y_var in y_variables:
            time_samp = 'M'
            df = mo[(mo['Crop'] == crop_name) & (mo['targetVar'] == y_var) & (mo['Time_sampling'] == time_samp)]
            # get the up2timeID
            pctPeak = []
            pctNull = []
            nTot = []
            xVals = np.sort(df[var4time].unique())
            for x in xVals:
                df_up2timeID = df[df[var4time] == x]
                # rank them
                df_up2timeID['Percentile_Rank'] = df_up2timeID[metric2use].rank(ascending=False, pct = True)
                df_up2timeID['Percentile_Rank'] = df_up2timeID['Percentile_Rank']*100
                nTot.append(df_up2timeID.shape[0])
                if crop_name == 'Soft wheat' and x == 8:
                    print('debug')
                pctPeak.append(df_up2timeID[df_up2timeID['Estimator']=='PeakNDVI']['Percentile_Rank'].values[0])
                pctNull.append(df_up2timeID[df_up2timeID['Estimator'] == 'Null_model']['Percentile_Rank'].values[0])
            axs.plot(up2Date, pctNull, color='grey', linewidth=1, marker='o', linestyle= lineStyle[cc], label='Null model '+ crop_name)
            axs.plot(up2Date, pctPeak, color='red', linewidth=1, marker='o', linestyle= lineStyle[cc], label='Peak NDVI ' + crop_name)
            cc = cc + 1

    axs.xaxis.set_major_formatter(fmt)
    axs.set_xlim(xlim[0], xlim[1])

    #lbl = r'${\rm Percentile\/of\/rRMSE_p\/ distribution\/(\%)}$'
    lbl = 'Percentile of\n' + r'${\rm rRMSE_p\/ distribution\/(\%)}$'
    axs.set_ylabel(lbl)
    axs.set_xlabel(xlabel)
    #axs.set_title(crop_name, fontsize=12)
    axs.legend(frameon=False, loc='lower left')
    axs.set_ylim(0.0, 100)
    strFn = os.path.join(dir, 'percentile_rank_peak_null_all_crops_' + y_var + '_by_time_sampling.png') #dir + '/' + 'percentile_rank_peak_null_all_crops_' + y_var + '_by_time_sampling.png'
    fig.savefig(strFn)
    #fig.savefig(strFn.replace(" ", ""))
    plt.close(fig)

    # Now for the best model and benchmark make scatter plot
    # teh refrence data is b1
    print('Compare output ended')


