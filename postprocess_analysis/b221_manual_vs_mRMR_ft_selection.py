import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import b05_Init
import datetime
from string import digits
import ast
import seaborn as sns


# her we try to disentangle effect of manual and auto ft selectio

def analyse_manual_vs_auto_feat_sel(dir, target):
    # Here we look at the effect of feature selection (OHE always on)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 400)  # width is = 400
    metric2use = 'rRMSE_p'#'RMSE_p' #'R2_p'
    if metric2use == 'R2_p':
        sortAscending = False
    elif metric2use == 'RMSE_p':
        sortAscending = True
    elif metric2use == 'rRMSE_p':
        sortAscending = True
    else:
        print('The metric is not coded, the function cannot be executed')
        return -1

    mo = pd.read_csv(dir + '/' + 'all_model_output.csv')


    # effect of only manual selection
    # delta rRMSEp between all variables (RS&Met and reduced subsets

    # For ft selection, focus on runs with OHE
    #Exclude benchmarks and no oHE
    moWB = mo[(mo['Estimator'] != 'Null_model') & (mo['Estimator'] != 'PeakNDVI') & (mo['DoOHEnc'] == 'AU_level')]
    print(moWB['Estimator'].unique())
    # attribute the column "Features" to a var set
    # remove OHE vars
    moWB['feat_set'] = moWB['Features'].map(lambda x: [y for y in ast.literal_eval(x) if not 'OHE' in y])
    # remove digits
    remove_digits = str.maketrans('', '', digits)
    moWB['feat_set'] = moWB['feat_set'].map(lambda x: [y.translate(remove_digits)[0:-1] for y in x])
    # get uniques
    moWB['feat_set'] = moWB['feat_set'].map(lambda x: list(set(x)))
    # give name to var set
    varSetDict = {
        'RS&Met': ['ND', 'NDmax', 'Rad', 'RainSum', 'T', 'Tmin', 'Tmax'],
        'RS&Met-': ['ND', 'RainSum', 'T'],
        'RS': ['ND', 'NDmax'],
        'RS-': ['ND'],
        'Met': ['Rad', 'RainSum', 'T', 'Tmin', 'Tmax'],
        'Met-': ['Rad', 'RainSum', 'T']
    }
    moWB['feat_set'] = moWB['feat_set'].map(lambda x: 'RS&Met' if set(x) == set(varSetDict['RS&Met']) else x)
    moWB['feat_set'] = moWB['feat_set'].map(lambda x: 'RS&Met-' if set(x) == set(varSetDict['RS&Met-']) else x)
    moWB['feat_set'] = moWB['feat_set'].map(lambda x: 'RS' if set(x) == set(varSetDict['RS']) else x)
    moWB['feat_set'] = moWB['feat_set'].map(lambda x: 'RS-' if set(x) == set(varSetDict['RS-']) else x)
    moWB['feat_set'] = moWB['feat_set'].map(lambda x: 'Met' if set(x) == set(varSetDict['Met']) else x)
    moWB['feat_set'] = moWB['feat_set'].map(lambda x: 'Met-' if set(x) == set(varSetDict['Met-']) else x)
    # assign a feature set order ascending by number of variables
    d = {'RS&Met': 1,
         'RS&Met-': 3,
         'RS': 5,
         'RS-': 6,
         'Met': 2,
         'Met-': 4
         }
    moWB['feat_set_order'] = moWB['feat_set'].map(d)


    # for each model configuration get results without ft_sel
    moWBWFTS = moWB[moWB['Ft_selection'] == 'none'].copy()
    # get results with RS& Met
    moWBWFTS_RSMET = moWBWFTS[moWBWFTS['feat_set'] == 'RS&Met']
    # Get results with reduced ft sets
    moWBWFTS_reducedSets =  moWBWFTS[moWBWFTS['feat_set'] != 'RS&Met']
    # join (at each ft set report RS&Met)
    col4join = ['forecast_time', 'targetVar', 'Crop', 'Estimator']
    colToTake = col4join + [metric2use] + ['feat_set', 'feat_set_order']
    moJoined = pd.merge(moWBWFTS_reducedSets, moWBWFTS_RSMET[colToTake], \
                        how='left', left_on=col4join, \
                        right_on=col4join)
    metric_RSMET = metric2use + '_RSMET'
    metric_reducedSets = metric2use + '_reducedSets'
    moJoined.rename(columns={metric2use + '_x': metric_reducedSets}, inplace=True)
    moJoined.rename(columns={metric2use + '_y': metric_RSMET}, inplace=True)
    # make the diff
    #compute the difference in performace for each modle run
    if metric2use == 'R2_p':
        moJoined[metric2use+'_improvementByManFtSet'] = (moJoined[metric_RSMET] - moJoined[metric_reducedSets]) #/ moJoined[metricWithoutFS_column_name]*100
    elif metric2use == 'RMSE_p' or metric2use =='rRMSE_p':
        moJoined[metric2use+'_improvementByManFtSet'] = (moJoined[metric_RSMET] - moJoined[metric_reducedSets]) #/ moJoined[metricWithoutFS_column_name]*100
    else:
        print('The metric is not coded, the function cannot be executed')
        return -1

    # get timing of forecasts
    project = b05_Init.init(target)
    # get the avg pheno
    pheno_avg = pd.read_csv(project['output_dir'] + '/' + project['AOI'] + '_pheno_mean_used.csv')
    up2month = pheno_avg[pheno_avg['Stage'] == 'sos']['Month'].to_numpy() + np.sort(moJoined['forecast_time'].unique())
    def progMonth2Month(x):
        if x > 12:
            x = x - 12
        return x
    up2month = list(map(progMonth2Month, up2month))
    up2Date = [datetime.datetime(2000, x, 1) for x in up2month]
    # reorder years
    up2Date_dYear = np.zeros(len(up2Date)).astype(int)
    # Find timing that are later than last time, and must go the year before
    sub = [i for i, x in enumerate(up2Date) if x > up2Date[-1]]
    up2Date_dYear[sub] = -1
    up2Date = [datetime.date(x.year + y, x.month, x.day) for x, y in zip(up2Date, up2Date_dYear)]
    monthOfForc = [x.strftime('%b-%d') for x in up2Date]
    #count % of times ft selection has a postive impact
    countPos = len(moJoined[moJoined[metric2use + '_improvementByManFtSet'] > 0])
    countTot = moJoined.shape[0]
    print('% ft has a positive effect')
    print(countPos/countTot*100.0)

    # Plot by Reduced feature set on X
    sns.boxplot(x="feat_set_order_x", y=metric2use + '_improvementByManFtSet',
                hue="forecast_time", palette="Blues", #ax=gs[0],
                data=moJoined,  linewidth=0.5, showfliers=False,)
    lbl = r'${\rm rRMSE_p\/RS&Met\/-rRMSE_p\/feature\/set\/(\%)}$'
    plt.ylabel(lbl)
    ll = [list(d.keys())[list(d.values()).index(x)] for x in [2, 3, 4, 5, 6]]
    plt.xlim(-1.5, 4.5)
    plt.xticks([0, 1, 2, 3, 4], ll)
    plt.xlabel('Feature set')
    L = plt.legend(loc="upper center", fontsize='x-small', labelspacing=0.1, columnspacing=0.5, ncol=8, frameon=False,
                   handlelength=1, title='Forecasting time',
                   title_fontsize='small')  # , labels=monthOfForc)  # , bbox_to_anchor=(0, 0.5, 0.5, 0.5)) ncol=8,
    for i in range(8):
        L.get_texts()[i].set_text(monthOfForc[i])
    # L.get_texts()[0].set_text('make it short')
    plt.axhline(0, color='black', linewidth=1)
    plt.ylim(-15, 15)
    plt.tight_layout()

    strFn = dir + '/' + 'Effect_of_Manual_selection_of_ft_set.png'
    plt.savefig(strFn.replace(" ", ""))
    #plt.show()
    plt.close()




    #now only mRMRM ft selection on Met&Rs
    #retain only Rs&Met
    moWB_RSMET = moWB[moWB['feat_set'] == 'RS&Met'].copy()
    # for each model configuration get results without ft_sel
    RSMET_withoutFtSel = moWB_RSMET[moWB_RSMET['Ft_selection'] == 'none'].copy()
    RSMET_withFtSel = moWB_RSMET[moWB_RSMET['Ft_selection'] == 'MRMR'].copy()
    # join
    col4join = ['forecast_time', 'targetVar', 'Crop', 'Estimator']
    colToTake = col4join + [metric2use] + ['feat_set', 'feat_set_order']
    moJoined = pd.merge(RSMET_withFtSel, RSMET_withoutFtSel[colToTake], \
                        how='left', left_on=col4join, \
                        right_on=col4join)
    metric_RSMET = metric2use + '_RSMET'
    metric_FtSel = metric2use + '_FtSel'
    moJoined.rename(columns={metric2use + '_x': metric_FtSel}, inplace=True)
    moJoined.rename(columns={metric2use + '_y': metric_RSMET}, inplace=True)
    # make the diff
    # compute the difference in performace for each modle run
    if metric2use == 'R2_p':
        moJoined[metric2use + '_improvementByMRMRFtSet'] = (moJoined[metric_RSMET] - moJoined[
            metric_FtSel])  # / moJoined[metricWithoutFS_column_name]*100
    elif metric2use == 'RMSE_p' or metric2use == 'rRMSE_p':
        moJoined[metric2use + '_improvementByMRMRFtSet'] = (moJoined[metric_RSMET] - moJoined[
            metric_FtSel])  # / moJoined[metricWithoutFS_column_name]*100
    else:
        print('The metric is not coded, the function cannot be executed')
        return -1

    # Plot by Reduced feature set on X
    sns.boxplot(x="feat_set_x", y=metric2use + '_improvementByMRMRFtSet',
                hue="forecast_time", palette="Blues",  # ax=gs[0],
                data=moJoined, linewidth=0.5, showfliers=False, )
    lbl = r'${\rm rRMSE_p\/without\/f.s.\/-\/rRMSE_p\/with\/f.s.\/(\%)}$'
    plt.ylabel(lbl)

    #ll = [list(d.keys())[list(d.values()).index(x)] for x in [2, 3, 4, 5, 6]]
    plt.xlabel('Feature set')
    L = plt.legend(loc="upper center", fontsize='x-small', labelspacing=0.1, columnspacing=0.5, ncol=8,
                   frameon=False,
                   handlelength=1, title='Forecasting time',
                   title_fontsize='small')  # , labels=monthOfForc)  # , bbox_to_anchor=(0, 0.5, 0.5, 0.5)) ncol=8,
    #plt.show()
    for i in range(8):
        L.get_texts()[i].set_text(monthOfForc[i])
    # L.get_texts()[0].set_text('make it short')
    plt.axhline(0, color='black', linewidth=1)
    plt.ylim(-15,15)

    strFn = dir + '/' + 'Effect_of_mRMR_selection.png'
    plt.savefig(strFn.replace(" ", ""))
    # plt.show()
    plt.close()






    # Plot by improvement with respect to RSMET of both manual and auto ft selection  FEATURE SET on X

    # for each model configuration get results without ft_sel
    # get results with RS&Met and no ft selection (the comparison term)
    moWB_withoutFtSel = moWB[moWB['Ft_selection'] == 'none'].copy()
    moWB_withoutFtSel = moWB_withoutFtSel[moWB_withoutFtSel['feat_set'] == 'RS&Met']
    # get all feat set wit ft selection
    moWB_withFtSel = moWB[moWB['Ft_selection'] == 'MRMR'].copy()

    # join (at each ft set report RS&Met)
    col4join = ['forecast_time', 'targetVar', 'Crop', 'Estimator']
    colToTake = col4join + [metric2use] + ['feat_set', 'feat_set_order']
    moJoined = pd.merge(moWB_withFtSel, moWB_withoutFtSel[colToTake], \
                        how='left', left_on=col4join, \
                        right_on=col4join)
    metric_RSMET = metric2use + '_RSMET'
    metric_ft_set_ft_sel = metric2use + '_ft_set_ft_sel'
    moJoined.rename(columns={metric2use + '_x': metric_ft_set_ft_sel}, inplace=True)
    moJoined.rename(columns={metric2use + '_y': metric_RSMET}, inplace=True)
    # make the diff
    # compute the difference in performace for each modle run
    if metric2use == 'R2_p':
        moJoined[metric2use + '_improvementByft_set_ft_sel'] = (moJoined[metric_RSMET] - moJoined[
            metric_ft_set_ft_sel])  # / moJoined[metricWithoutFS_column_name]*100
    elif metric2use == 'RMSE_p' or metric2use == 'rRMSE_p':
        moJoined[metric2use + '_improvementByft_set_ft_sel'] = (moJoined[metric_RSMET] - moJoined[
            metric_ft_set_ft_sel])  # / moJoined[metricWithoutFS_column_name]*100
    else:
        print('The metric is not coded, the function cannot be executed')
        return -1
    # plt.figure(figsize=(6.4, 4.8))
    plt.figure(figsize=(1.75, 4.8))
    sns.boxplot(x="feat_set_order_x", y=metric2use + '_improvementByft_set_ft_sel',
                hue="forecast_time", palette="Blues",  # ax=gs[0],
                data=moJoined, linewidth=0.5, showfliers=False)  # , labels=["1a","2a","3a","4a","5a"]
    lbl = r'${\rm rRMSE_p\/RS&Met\/without\/f.s.\/-\/rRMSE_p\/with\/f.s.\/(\%)}$'
    plt.ylabel(lbl)
    ll = [list(d.keys())[list(d.values()).index(x)] for x in [1, 2, 3, 4, 5, 6]]
    plt.xticks([0, 1, 2, 3, 4, 5], ll)
    plt.xlabel('Feature set')
    plt.legend([], [], frameon=False)
    # L = plt.legend(loc="upper center", fontsize='x-small', labelspacing=0.1, columnspacing=0.5, ncol=8, frameon=False,
    #                handlelength=1, title='Forecasting time',
    #                title_fontsize='small')  # , labels=monthOfForc)  # , bbox_to_anchor=(0, 0.5, 0.5, 0.5)) ncol=8,
    # for i in range(8):
    #     L.get_texts()[i].set_text(monthOfForc[i])
    # # L.get_texts()[0].set_text('make it short')
    plt.axhline(0, color='black', linewidth=1)
    plt.ylim(-15, 15)
    plt.xlim(-0.5, 0.5)
    plt.tight_layout()
    #fig_width, fig_height = plt.gcf().get_size_inches()
    strFn = dir + '/' + 'Effect_of_feature_selection_by_FeatSetANDforecast_time_respect_to_RSeMet_set1' \
                        '.png'
    plt.savefig(strFn.replace(" ", ""))

    #plt.figure(figsize=(6.4, 4.8)) #this is standard
    plt.figure(figsize=(5.2, 4.8))
    sns.boxplot(x="feat_set_order_x", y=metric2use + '_improvementByft_set_ft_sel',
                hue="forecast_time", palette="Blues",  # ax=gs[0],
                data=moJoined, linewidth=0.5, showfliers=False)  # , labels=["1a","2a","3a","4a","5a"]
    lbl = r'${\rm rRMSE_p\/RS&Met\/without\/f.s.\/-\/rRMSE_p\/with\/f.s.\/(\%)}$'
    plt.ylabel(lbl)
    ll = [list(d.keys())[list(d.values()).index(x)] for x in [1, 2, 3, 4, 5, 6]]
    plt.xticks([0, 1, 2, 3, 4, 5], ll)
    plt.xlabel('Feature set')
    L = plt.legend(loc="upper center", fontsize='x-small', labelspacing=0.1, columnspacing=0.5, ncol=8, frameon=False,
                   handlelength=1, title='Forecasting time',
                   title_fontsize='small')  # , labels=monthOfForc)  # , bbox_to_anchor=(0, 0.5, 0.5, 0.5)) ncol=8,
    for i in range(8):
        L.get_texts()[i].set_text(monthOfForc[i])
    # L.get_texts()[0].set_text('make it short')
    plt.axhline(0, color='black', linewidth=1)
    plt.ylim(-15, 15)
    plt.xlim(0.5, 5.5)
    plt.tight_layout()
    strFn = dir + '/' + 'Effect_of_feature_selection_by_FeatSetANDforecast_time_respect_to_RSeMet_set2' \
                        '.png'
    plt.savefig(strFn.replace(" ", ""))

    # plt.figure(figsize=(6.4, 4.8))
    # sns.boxplot(x="feat_set_order_x", y=metric2use + '_improvementByft_set_ft_sel',
    #             hue="forecast_time", palette="Blues",  # ax=gs[0],
    #             data=moJoined, linewidth=0.5, showfliers=False)  # , labels=["1a","2a","3a","4a","5a"]
    # lbl = r'${\rm rRMSE_p\/RS&MET\/without\/f.s.\/-\/rRMSE_p\/with\/f.s.\/(\%)}$'
    # plt.ylabel(lbl)
    # ll = [list(d.keys())[list(d.values()).index(x)] for x in [1, 2, 3, 4, 5, 6]]
    # plt.xticks([0, 1, 2, 3, 4, 5], ll)
    # plt.xlabel('Feature set')
    # L = plt.legend(loc="upper center", fontsize='x-small', labelspacing=0.1, columnspacing=0.5, ncol=8, frameon=False,
    #                handlelength=1, title='Forecasting time',
    #                title_fontsize='small')  # , labels=monthOfForc)  # , bbox_to_anchor=(0, 0.5, 0.5, 0.5)) ncol=8,
    # for i in range(8):
    #     L.get_texts()[i].set_text(monthOfForc[i])
    # # L.get_texts()[0].set_text('make it short')
    # plt.axhline(0, color='black', linewidth=1)
    # plt.ylim(-15, 15)
    # plt.tight_layout()
    # strFn = dir + '/' + 'Effect_of_feature_selection_by_FeatSetANDforecast_time_respect_to_RSeMet_set3' \
    #                     '.png'
    # plt.savefig(strFn.replace(" ", ""))

    # plt.show()
    plt.close()
    print('ended')


res = analyse_manual_vs_auto_feat_sel('X:/PY_data/ML1_data_output/Algeria/Final_run/20210219','Algeria')