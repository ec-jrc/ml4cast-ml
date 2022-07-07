import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import b05_Init
import datetime
from string import digits
import ast
import seaborn as sns
import pathlib

def analyse_feat_sel(dir, target):
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

    # for a given y_var, all lead times, all crops, all ML models (thus excluding benchmarks), and with OHE on we look
    # at the increase of performances related to the use of feat selection (yes vs no) by feature set
    # Then we plot c = % of feature sel y = increase of perf using different line for different feature sets.

    # For ft selection, focus on runs with OHE
    #Exclude benchmarks and no oHE
    moOheWithoutBench = mo[(mo['Estimator'] != 'Null_model') & (mo['Estimator'] != 'PeakNDVI') & (mo['DoOHEnc'] == 'AU_level')]
    print(moOheWithoutBench['Estimator'].unique())
    #Get model with feat sel
    moWithFS = moOheWithoutBench[moOheWithoutBench['Ft_selection'] == 'MRMR'].copy()
    # Get model without feat sel
    moWithoutFS = moOheWithoutBench[moOheWithoutBench['Ft_selection'] == 'none'].copy()
    # Join the RMSEp of without to the with data frame
    col4join = ['forecast_time','N_features', 'N_OHE','Features','targetVar','Crop','Estimator']
    colToTake = col4join + [metric2use]
    moJoined = pd.merge(moWithFS, moWithoutFS[colToTake],\
            how='left', left_on=col4join, \
                        right_on=col4join)
    metricWithoutFS_column_name = metric2use+'_WithoutFS'
    metricWithFS_column_name = metric2use + '_WithFS'
    moJoined.rename(columns={metric2use+'_y': metricWithoutFS_column_name}, inplace=True)
    moJoined.rename(columns={metric2use + '_x': metricWithFS_column_name}, inplace=True)
    #moJoined.to_csv(dir + '/' + 'buttami.csv', index=False)


    # attribute the column "Features" to a var set
    # remove OHE vars
    moJoined['feat_set'] = moJoined['Features'].map(lambda x: [y for y in ast.literal_eval(x) if not 'OHE' in y])
    # remove digits
    remove_digits = str.maketrans('', '', digits)
    moJoined['feat_set'] = moJoined['feat_set'].map(lambda x: [y.translate(remove_digits)[0:-1] for y in x])
    # get uniques
    moJoined['feat_set'] = moJoined['feat_set'].map(lambda x: list(set(x)))
    # give name to var set
    varSetDict = {
        'RS&Met': ['ND', 'NDmax', 'Rad', 'RainSum', 'T', 'Tmin', 'Tmax'],
        'RS&Met-': ['ND', 'RainSum', 'T'],
        'RS': ['ND', 'NDmax'],
        'RS-': ['ND'],
        'Met': ['Rad', 'RainSum', 'T', 'Tmin', 'Tmax'],
        'Met-': ['Rad', 'RainSum', 'T']
    }
    moJoined['feat_set'] = moJoined['feat_set'].map(lambda x: 'RS&Met' if set(x) == set(varSetDict['RS&Met']) else x)
    moJoined['feat_set'] = moJoined['feat_set'].map(lambda x: 'RS&Met-' if set(x) == set(varSetDict['RS&Met-']) else x)
    moJoined['feat_set'] = moJoined['feat_set'].map(lambda x: 'RS' if set(x) == set(varSetDict['RS']) else x)
    moJoined['feat_set'] = moJoined['feat_set'].map(lambda x: 'RS-' if set(x) == set(varSetDict['RS-']) else x)
    moJoined['feat_set'] = moJoined['feat_set'].map(lambda x: 'Met' if set(x) == set(varSetDict['Met']) else x)
    moJoined['feat_set'] = moJoined['feat_set'].map(lambda x: 'Met-' if set(x) == set(varSetDict['Met-']) else x)
    #assign a feature set order ascending by number of variables
    d = {'RS&Met': 1,
        'RS&Met-': 3,
        'RS': 5,
        'RS-': 6,
        'Met': 2,
        'Met-': 4
    }
    moJoined['feat_set_order'] = moJoined['feat_set'].map(d)
    #compute the difference in performace for each modle run
    if metric2use == 'R2_p':
        moJoined[metric2use+'_improvementByFT'] = (moJoined[metricWithFS_column_name] - moJoined[metricWithoutFS_column_name]) #/ moJoined[metricWithoutFS_column_name]*100
    elif metric2use == 'RMSE_p' or metric2use =='rRMSE_p':
        moJoined[metric2use+'_improvementByFT'] = - (moJoined[metricWithFS_column_name] - moJoined[metricWithoutFS_column_name]) #/ moJoined[metricWithoutFS_column_name]*100
    else:
        print('The metric is not coded, the function cannot be executed')
        return -1
    #now compute avg and sd improvement bu feature set
    res = moJoined.groupby(['feat_set'], as_index=False).agg(
        {metric2use+'_improvementByFT': ['mean', 'std'], 'feat_set_order': 'first'}).sort_values(by=[('feat_set_order','first')], ascending=True)
    res2 = moJoined.groupby(['forecast_time'], as_index=False).agg(
        {metric2use + '_improvementByFT': ['mean', 'std']})
    res3 = moJoined.groupby(['Estimator'], as_index=False).agg(
        {metric2use + '_improvementByFT': ['mean', 'std']})
    res4 = moJoined.groupby(['feat_set','forecast_time'], as_index=False).agg(
        {metric2use + '_improvementByFT': ['mean', 'std']})
    res5 = moJoined.groupby(['Estimator', 'forecast_time'], as_index=False).agg(
        {metric2use + '_improvementByFT': ['mean', 'std'], 'Prct_selected_fit': ['mean']})
    #print(res5)

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
    countPos = len(moJoined[moJoined[metric2use + '_improvementByFT'] > 0])
    countTot = moJoined.shape[0]
    print('% ft has a positive effect')
    print(countPos/countTot*100.0)

    # Plot by ESTIMATOR on X
    sns.boxplot(x="Estimator", y=metric2use + '_improvementByFT',
                hue="forecast_time", palette="Blues", #ax=gs[0],
                data=moJoined,  linewidth=0.5, showfliers=False,)
    lbl = r'${\rm rRMSE_p\/without\/f.s.\/-\/rRMSE_p\/with\/f.s.\/(\%)}$'
    plt.ylabel(lbl)
    plt.xlabel('Algorithm')
    # adjust ticks
    currentTicks = plt.xticks()
    labels = [item.get_text() for item in currentTicks[1]]
    labels = [x.replace('_', ' ') for x in labels]
    labels = ['RF' if x == 'RandomForest' else x for x in labels]
    plt.xticks([0, 1, 2, 3, 4, 5], labels)
    L = plt.legend(loc="upper center", fontsize='x-small', labelspacing=0.1, columnspacing=0.5, ncol=8, frameon=False,
                   handlelength=1, title='Forecasting time',
                   title_fontsize='small')
    for i in range(8):
        L.get_texts()[i].set_text(monthOfForc[i])
    plt.axhline(0, color='black', linewidth=1)
    strFn = dir + '/' + 'Effect_of_feature_selection_by_estimatorANDforecast_time.png'
    plt.savefig(strFn.replace(" ", ""))
    #plt.show()
    plt.close()
    # Plot by ESTIMATOR on X and for all feature sets as suggested by Franz
    list_ft_set_order = sorted(d.values())
    for o in list_ft_set_order:
        print(o)
        moJoinedFtSet =  moJoined[moJoined['feat_set_order'] == o]
        ftSetName = list(d.keys())[list(d.values()).index(o)]
        sns.boxplot(x="Estimator", y=metric2use + '_improvementByFT',
                    hue="forecast_time", palette="Blues",  # ax=gs[0],
                    data=moJoinedFtSet, linewidth=0.5, showfliers=False, )
        lbl = r'${\rm rRMSE_p\/without\/f.s.\/-\/rRMSE_p\/with\/f.s.\/(\%)}$'
        plt.ylabel(lbl) #        plt.ylabel("rRMSEp without f.s. - rRMSEp with f.s. (%)")
        plt.xlabel('Algorithm')
        plt.title(ftSetName)
        L = plt.legend(loc="upper center", fontsize='x-small', labelspacing=0.1, columnspacing=0.5, ncol=8,
                       frameon=False,
                       handlelength=1, title='Forecasting time',
                       title_fontsize='small')
        # when rs-1 there is no ft selection at time 1, so adjust month of forcasts
        availmonthOfForc = [monthOfForc[x - 1] for x in sorted(moJoinedFtSet['forecast_time'].unique().tolist())]
        #if o == 6:
        #    print('debug')
        plt.ylim(-6, 18)
        for i in range(len(availmonthOfForc)):
            L.get_texts()[i].set_text(availmonthOfForc[i])
        plt.axhline(0, color='black', linewidth=1)
        strFn = dir + '/' + 'Effect_of_feature_selection_by_estimatorANDforecast_time_set_' + ftSetName + '.png'
        plt.savefig(strFn.replace(" ", ""))
        # plt.show()
        plt.close()




    # Plot by FEATURE SET on X
    sns.boxplot(x="feat_set_order", y=metric2use + '_improvementByFT',
                hue="forecast_time", palette="Blues",  # ax=gs[0],
                data=moJoined, linewidth=0.5, showfliers=False)  # , labels=["1a","2a","3a","4a","5a"]
    lbl = r'${\rm rRMSE_p\/without\/f.s.\/-\/rRMSE_p\/with\/f.s.\/(\%)}$'
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
    strFn = dir + '/' + 'Effect_of_feature_selection_by_FeatSetANDforecast_time.png'
    plt.savefig(strFn.replace(" ", ""))
    # plt.show()
    plt.close()



    # Now see effect of OHE
    # Exclude benchmarks
    moOheWithoutBench = mo[
        (mo['Estimator'] != 'Null_model') & (mo['Estimator'] != 'PeakNDVI')]
    # make a new column with variables used (without OHE). This is needed for the join, the number of var excluding OHE is not sufficient as 2 feature sets have the same number
    moOheWithoutBench['tmp'] = moOheWithoutBench['Features'].map(lambda x: [y for y in ast.literal_eval(x) if not 'OHE' in y])
    moOheWithoutBench['nonOHEfeatures']=  moOheWithoutBench['tmp'].map(lambda x: ','.join(map(str, x)))
                 #moOheWithoutBench['nonOHEfeatures'] = moOheWithoutBench['nonOHEfeatures'].astype(str)
    # Get model with OHE
    moWithOHE = moOheWithoutBench[moOheWithoutBench['DoOHEnc'] == 'AU_level']
    # Get model without feat sel
    moWithoutOHE = moOheWithoutBench[moOheWithoutBench['DoOHEnc'] == 'none']
    # Join the RMSEp of without to the with data frame
    col4join = ['forecast_time', 'nonOHEfeatures', 'Ft_selection', 'targetVar', 'Crop', 'Estimator']
    colToTake = col4join + [metric2use]

    moJoined = pd.merge(moWithOHE, moWithoutOHE[colToTake], \
                        how='left', left_on=col4join, \
                        right_on=col4join)
    metricWithoutOHE_column_name = metric2use + '_noOHE'
    metricWithOHE_column_name = metric2use + '_OHE'
    moJoined.rename(columns={metric2use + '_y': metricWithoutOHE_column_name}, inplace=True)
    moJoined.rename(columns={metric2use + '_x': metricWithOHE_column_name}, inplace=True)
    # compute the difference in performace for each model run
    if metric2use == 'R2_p':
        moJoined[metric2use + '_improvementByOHE'] = (moJoined[metricWithOHE_column_name] - moJoined[
            metricWithoutOHE_column_name])  # / moJoined[metricWithoutFS_column_name]*100
    elif metric2use == 'RMSE_p' or metric2use == 'rRMSE_p':
        moJoined[metric2use + '_improvementByOHE'] = - (moJoined[metricWithOHE_column_name] - moJoined[
            metricWithoutOHE_column_name])  # / moJoined[metricWithoutFS_column_name]*100
    else:
        print('The metric is not coded, the function cannot be executed')
        return -1

    res7 = moJoined.groupby(['forecast_time'], as_index=False).agg(
        {metric2use + '_improvementByOHE': ['mean', 'std']})


    sns.boxplot(x="forecast_time", y=metric2use + '_improvementByOHE',
                palette="Reds",  # ax=gs[0],
                data=moJoined, linewidth=0.5, showfliers=False)  # , labels=["1a","2a","3a","4a","5a"]
    #plt.ylabel("rRMSEp without OHE - rRMSEp with OHEau (%)")
    lbl = r'${\rm rRMSE_p\/without\/OHEau\/-\/rRMSE_p\/with\/OHEau\/(\%)}$'
    #lbl = r'${\rm rRMSE_p}$'
    plt.ylabel(lbl)
    plt.xlabel('Forecasting time')
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], monthOfForc)
    plt.axhline(0, color='black', linewidth=1)
    strFn = dir + '/' + 'Effect_OHE_by_forecast_time.png'
    plt.savefig(strFn.replace(" ", ""))
    # plt.show()
    plt.close()

    # Effect of OHE by methods
    sns.boxplot(x="Estimator", y=metric2use + '_improvementByOHE',
                hue="forecast_time", palette="Blues",  # ax=gs[0],
                data=moJoined, linewidth=0.5, showfliers=False, )
    lbl = r'${\rm rRMSE_p\/without\/OHEau\/-\/rRMSE_p\/with\/OHEau\/(\%)}$'
    plt.ylabel(lbl)

    plt.xlabel('Algorithm')
    L = plt.legend(loc="upper center", fontsize='x-small', labelspacing=0.1, columnspacing=0.5, ncol=8, frameon=False,
                   handlelength=1, title='Forecasting time',
                   title_fontsize='small')
    for i in range(8):
        L.get_texts()[i].set_text(monthOfForc[i])
    #adjust ticks
    currentTicks =  plt.xticks()
    labels = [item.get_text() for item in currentTicks[1]]
    labels = [x.replace('_', ' ') for x in labels]
    labels = ['RF' if x == 'RandomForest' else x for x in labels]
    plt.xticks([0, 1, 2, 3, 4, 5], labels)
    #plt.xticks([0, 1, 2, 3, 4, 5], ['Lasso', 'RF', 'SVR rbf', 'SVR linear', 'GBR', 'MLP'])
    plt.axhline(0, color='black', linewidth=1)
    strFn = dir + '/' + 'Effect_of_OHE_by_estimatorANDforecast_time.png'
    plt.savefig(strFn.replace(" ", ""))
    # plt.show()
    plt.close()

    #Effect of ft set by time (for ft selection yes no, and for each model)
    #Start from scratch
    # Exclude benchmarks and no oHE
    moOheWithoutBench = mo[
        (mo['Estimator'] != 'Null_model') & (mo['Estimator'] != 'PeakNDVI') & (mo['DoOHEnc'] == 'AU_level')]

    # attribute the column "Features" to a var set
    # remove OHE vars
    moOheWithoutBench['feat_set'] = moOheWithoutBench['Features'].map(lambda x: [y for y in ast.literal_eval(x) if not 'OHE' in y])
    # remove digits
    remove_digits = str.maketrans('', '', digits)
    moOheWithoutBench['feat_set'] = moOheWithoutBench['feat_set'].map(lambda x: [y.translate(remove_digits)[0:-1] for y in x])
    # get uniques
    moOheWithoutBench['feat_set'] = moOheWithoutBench['feat_set'].map(lambda x: list(set(x)))
    # give name to var set
    varSetDict = {
        'RS&Met': ['ND', 'NDmax', 'Rad', 'RainSum', 'T', 'Tmin', 'Tmax'],
        'RS&Met-': ['ND', 'RainSum', 'T'],
        'RS': ['ND', 'NDmax'],
        'RS-': ['ND'],
        'Met': ['Rad', 'RainSum', 'T', 'Tmin', 'Tmax'],
        'Met-': ['Rad', 'RainSum', 'T']
    }
    moOheWithoutBench['feat_set'] = moOheWithoutBench['feat_set'].map(lambda x: 'RS&Met' if set(x) == set(varSetDict['RS&Met']) else x)
    moOheWithoutBench['feat_set'] = moOheWithoutBench['feat_set'].map(lambda x: 'RS&Met-' if set(x) == set(varSetDict['RS&Met-']) else x)
    moOheWithoutBench['feat_set'] = moOheWithoutBench['feat_set'].map(lambda x: 'RS' if set(x) == set(varSetDict['RS']) else x)
    moOheWithoutBench['feat_set'] = moOheWithoutBench['feat_set'].map(lambda x: 'RS-' if set(x) == set(varSetDict['RS-']) else x)
    moOheWithoutBench['feat_set'] = moOheWithoutBench['feat_set'].map(lambda x: 'Met' if set(x) == set(varSetDict['Met']) else x)
    moOheWithoutBench['feat_set'] = moOheWithoutBench['feat_set'].map(lambda x: 'Met-' if set(x) == set(varSetDict['Met-']) else x)
    # assign a feature set order ascending by number of variables
    d = {'RS&Met': 1,
         'RS&Met-': 3,
         'RS': 5,
         'RS-': 6,
         'Met': 2,
         'Met-': 4
         }
    # attach ft set order
    moOheWithoutBench['feat_set_order'] = moOheWithoutBench['feat_set'].map(d)
    # Create an array with the colors you want to use
    RM = "#cc0000"
    RMminus = "#ff9999"
    M = "#3333ff"
    Mminus = "#99ccff"
    R = "#ffff00"
    Rminus = "#ffff99"
    colors = [RM, M, RMminus, Mminus, R, Rminus]
    # Set your custom color palette
    customPalette = sns.set_palette(sns.color_palette(colors))

    # analysie without an with ft sel
    for ftSel in ['none']: #, 'MRMR']:
        df =  moOheWithoutBench[moOheWithoutBench['Ft_selection'] == ftSel]
        # now model by model
        modelList = df['Estimator'].unique().tolist()
        modelList.append('All ML models')
        for model in modelList:
            if model == 'All ML models':
                dfMod = df
            else:
                dfMod = df[df['Estimator'] == model]
            sns.boxplot(x="forecast_time", y=metric2use,
                        hue="feat_set_order", palette=customPalette,  # ax=gs[0],
                        data=dfMod, linewidth=0.5, showfliers=False, )
            lbl = r'${\rm rRMSE_p\/(\%)}$'
            plt.ylabel(lbl)
            #plt.ylabel("rRMSEp (%)")
            plt.xlabel('Forecasting time')
            L = plt.legend(loc="upper center", fontsize='x-small', labelspacing=0.1, columnspacing=0.5, ncol=8,
                           frameon=False,
                           handlelength=1, title='Feature set',
                           title_fontsize='small')
            #ftSetList = [monthOfForc[x - 1] for x in sorted(moJoinedFtSet['forecast_time'].unique().tolist())]
            ll = [list(d.keys())[list(d.values()).index(x)] for x in [1, 2, 3, 4, 5, 6]]

            for i in range(len(ll)):
                #L.get_texts()[i].set_text(monthOfForc[i])
                L.get_texts()[i].set_text(ll[i])
            #plt.axhline(0, color='black', linewidth=1)
            plt.title(model)
            plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], monthOfForc)
            strFn = dir + '/' + 'Effect_of_Feature_Set_OHEon_model_' + model + '_ftSe_' + ftSel + '_by_time.png'
            plt.savefig(strFn.replace(" ", ""))
            # plt.show()
            plt.close()

    print('ended')






res = analyse_feat_sel('X:/PY_data/ML1_data_output/Algeria/Final_run/20210219','Algeria')
