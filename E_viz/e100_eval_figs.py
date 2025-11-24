import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import os
import pathlib
import calendar
from A_config import a10_config
from D_modelling import d090_model_wrapper, d140_modelStats
from D_modelling import d140_modelStats
from E_viz import e50_yield_data_analysis
def addStringIfNotEmpty(info, x, sep = ','):
    if x != '':
        if len(info)>0:
            info = info + sep +'  ' + str(x)
        else:
            info = str(x)
    return info
def output_row_to_ML_info_string(df, metric2use):
    info_string0 = str(round(df[metric2use].values[0],2))
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
    info_string3 = addStringIfNotEmpty(info_string3, fs)
    prct= round(df.Prct_selected_fit.values[0]) if fs != '' else ''
    info_string3 = addStringIfNotEmpty(info_string3, prct, sep=':')
    return [info_string0, info_string1, info_string2, info_string3]


def AU_error(b1, config, outputDir, suffix, adm_id_in_shp_2keep=None):
    # Starting from b1 (best 1 model, avg statistics)

    # In this version I compute the rel RMSE by admin (each admin with its own mean yield) and then I weight them
    # using area of the last five years (to occount for the fact that larger errors are more tolerable if the area is small)
    # 2024/12/19 note: the last five years has problem in Harvest data (e.g. Zambia) where some units do not have the last 5 yrs,
    # therefore I use the full time series stats

    mlsettings = a10_config.mlSettings(forecastingMonths=0)

    os.path.join(config.data_dir, 'Label_analysis')
    df_Stats = pd.read_csv(os.path.join(os.path.join(config.data_dir, 'Label_analysis' + str(config.prct2retain)),
                                            config.AOI + '_LTstats_retainPRCT' + str(config.prct2retain) + '.csv'))
    df_regNames = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_REGION_id.csv'))
    # Now read mres, compute metric2use at the admin level, and make an area average add as a new columns
    b1['rRMSE_p_areaWeighted'] = -999
    b1 = b1.reset_index()  # the index was repeating
    dfAU = pd.DataFrame()

    # for each best model and benchmarks, compute error at the admin level using mRes
    for index, row in b1.iterrows():
        # get run_id
        runID = row['runID']
        myID = f'{runID:06d}'
        # build the mRes file name, read it if present, mae it if not present
        fn_mRes_out = os.path.join(config.models_out_dir, 'ID_' + str(myID) +
                                   '_crop_' + row['Crop'] + '_Yield_' + row['Estimator'] +
                                   '_mres.csv')
        if os.path.exists(fn_mRes_out):
            mRes = pd.read_csv(fn_mRes_out)
        else:
            fn_spec = os.path.join(config.models_spec_dir, str(myID) + '_' + row['Crop'] + '_' + row['Estimator'] + '.json')
            mRes = d090_model_wrapper.fit_and_validate_single_model(fn_spec, config, 'tuning', run2get_mres_only=True)

        # if there was a crop_au_exclusions for ML, remove results for all (they are there for benchmark) to make a fair
        # comparison
        if bool(config.crop_au_exclusions):
            if row['Crop'] in config.crop_au_exclusions.keys():
                # print(row['Estimator'])
                au_id_to_remove = df_regNames[df_regNames['adm_name'].isin(config.crop_au_exclusions[row['Crop']])]['adm_id'].tolist()
                mRes = mRes.loc[~mRes['adm_id'].isin(au_id_to_remove)]

        rRMSE_pByAdmin = d140_modelStats.statsByAdmin(mRes)
        rRMSE_pByAdmin = rRMSE_pByAdmin.merge(df_regNames, how='left', left_on='adm_id', right_on='adm_id')
        rRMSE_pByAdmin = rRMSE_pByAdmin.merge(df_Stats[df_Stats['Crop_name|first'] == row['Crop']], how='left',
                                              left_on='adm_id', right_on='adm_id|')
        # The wighted avg was computed using au rrmse, which is not coorect, It has to be computed using all data at once,
        # weighting each single error-admin by the area-admin

        mResWithArea = mRes.merge(rRMSE_pByAdmin[['adm_id|', 'Area|mean']], how='left', left_on='adm_id', right_on='adm_id|')
        z = d140_modelStats.rmse_rrmse_weighed_overall(mRes, mResWithArea['Area|mean'])
        row['rRMSE_p_areaWeighted'] = z['rel_Pred_RMSE']
        rRMSE_pByAdmin.insert(1, column='Estimator', value=row['Estimator'])
        rRMSE_pByAdmin.insert(2, column='forecast_time', value=row['forecast_time'])
        rRMSE_pByAdmin = rRMSE_pByAdmin.assign(**row.to_frame().T.to_dict(orient='records')[0])

        dfAU = pd.concat([dfAU, rRMSE_pByAdmin])
    dfAU.to_csv(os.path.join(outputDir, 'all_model_best1_AU_error.csv'))
    crops = dfAU['Crop'].unique()
    # exclude admins based on rRMSEp threshold on the first forecast
    if True:
        rrmse_prct_threshold = 50
        first_forecast_time = dfAU['forecast_time'].min()
        dfExc = dfAU.loc[dfAU['forecast_time'] == first_forecast_time, :]
        stringOut = '"crop_au_exclusions": {'
        for crop in crops:
            dfExcCrop = dfExc[dfExc['Crop'] == crop].copy()
            # keep ML only
            dfExcCrop = dfExcCrop[~dfExcCrop['Estimator'].isin(['Trend', 'PeakNDVI', 'Null_model'])]
            dfExcCrop = dfExcCrop[dfExcCrop['rrmse_prct'] > rrmse_prct_threshold]
            # out: "crop_au_exclusions": {"Sunflower": ["Northern Cape"]},
            lst2excl = dfExcCrop['adm_name'].unique().tolist()
            s = '[' + ', '.join(
                '"' + elem + '"' for elem in
                lst2excl) + ']'
            stringOut = stringOut + '"' + crop + '": ' + s + ", "
        #remove final comma and finalize the line
        stringOut = stringOut.rstrip(', ') + "}"
        with open(os.path.join(outputDir, 'rRMSEpGT' + str(rrmse_prct_threshold) + '_exclusion_string.txt'), "w") as file:
            file.write(stringOut)
    if not adm_id_in_shp_2keep is None:
        # boundaries are changing in time (Morocco case), keep only admin represented in the last shp
        dfAU = dfAU[dfAU['adm_id'].isin(adm_id_in_shp_2keep)]
        dfAU.to_csv(os.path.join(outputDir, 'all_model_best1_AU_error' + suffix + '.csv'))
    # Plot
    for crop in crops:
        dfAUc = dfAU[dfAU['Crop'] == crop].copy()
        forcTime = dfAUc["forecast_time"].unique()

        fig, axes = plt.subplots(len(forcTime), 1, figsize=(8*len(forcTime),10))
        ax_c = 0
        for ftime in forcTime:
            tmp = dfAUc[dfAUc["forecast_time"] == ftime]
            tmp = tmp.sort_values('adm_name').reset_index()
            # to assign constant colors, find the name of the ML estimator
            #ml_est_name = np.setdiff1d(list(tmp['Estimator'].unique()), ['Trend', 'PeakNDVI', 'Null_model'])[0]
            ml_est_name = np.setdiff1d(list(tmp['Estimator'].unique()), mlsettings.benchmarks)[0]
            # palette = {"Trend": "g", "PeakNDVI": "r", "Null_model": "grey", ml_est_name: "b"}
            # Tab change 2025
            palette = {"Trend": "g", "PeakNDVI": "r", "Null_model": "grey", "Tab": "purple", ml_est_name: "b"}
            # limit string of adm name
            tmp["adm_name_short"] = tmp["adm_name"].astype(str).str[:15]
            p1 = sns.barplot(tmp, x="adm_name_short", y="rrmse_prct", hue="Estimator", ax=axes[ax_c], palette=palette, order=tmp['adm_name_short'])
            p1.set_title('Forecast_time = ' + str(ftime))
            axes[ax_c].set_ylim(0, 100)
            sns.move_legend(axes[ax_c], "upper right", title=None, frameon=False) #bbox_to_anchor=(1, 1)
            # plt.xticks(rotation=70)
            p1.set_xticklabels(p1.get_xticklabels(),
                                      rotation=70,
                                      horizontalalignment='right')
            ax_c = ax_c + 1
            # if len(forcTime) > 1:
            #     tmp = dfAUc[dfAUc["forecast_time"] == forcTime[1]]
            #     tmp = tmp.sort_values('adm_name').reset_index()
            #     # ml_est_name = np.setdiff1d(list(tmp['Estimator'].unique()), ['Trend', 'PeakNDVI', 'Null_model'])[0]
            #     ml_est_name = np.setdiff1d(list(tmp['Estimator'].unique()), mlsettings.benchmarks)[0]
            #     # Tab change 2025
            #     palette = {"Trend": "g", "PeakNDVI": "r", "Null_model": "grey", "Tab": "purple", ml_est_name: "b"}
            #     # palette = {"Trend": "g", "PeakNDVI": "r", "Null_model": "grey", ml_est_name: "b"}
            #     p2 = sns.barplot(tmp, x="adm_name", y="rrmse_prct", hue="Estimator", ax=axes[1], palette=palette, order=tmp['adm_name'])
            #     p2.set_title('Forecast_time = ' + str(forcTime[1]))
            #     sns.move_legend(axes[1], "upper right", title=None, frameon=False)  # bbox_to_anchor=(1, 1)
            #     # plt.xticks(rotation=70)
            #     p2.set_xticklabels(p1.get_xticklabels(),
            #                        rotation=70,
            #                        horizontalalignment='right')

        text = ''
        if bool(config.crop_au_exclusions):
            if crop in config.crop_au_exclusions.keys():
                text = ', Ml omitting: ' + ",".join(config.crop_au_exclusions[crop])
        plt.suptitle(crop + text)
        plt.tight_layout()
        plt.savefig(os.path.join(outputDir, 'all_model_best1_AU_error_' + crop + suffix +'.png'))
        plt.close(fig)
    return dfAU
def bars_by_forecast_time2(b1, config, metric2use, mlsettings, var4time, outputDir):
    # In this version I compute the rel RMSE by admin (aeach admin with its own mean yield) and then I weight them
    # using area of the last five years (to occount for the fact that larger errors are more tolerable if the area is small)

    # get overall stats (dropping AU duplicated, here I have for the same run ID results for each AU, here
    # I am interested only in overall results, that is the same for all duplicates)
    b1 = b1.drop_duplicates(subset='runID', keep='first')
    # Tab change 2025
    # colors = {'ML': "#0000FF", 'Null_model': "#969696", 'PeakNDVI': "#FF0000", 'Trend': "#009600", 'BestByAdmin': "#FFFF00"}
    colors = {'ML': "#0000FF", 'Null_model': "#969696", 'PeakNDVI': "#FF0000", 'Trend': "#009600", 'Tab': "#660066",
              'BestByAdmin': "#FFFF00"}
    for t in b1[var4time].unique():
        crops = b1['Crop'].unique()
        # get forecast_issue_calendar_month
        forecast_issue_calendar_month = calendar.month_abbr[
            b1[b1[var4time] == t]['forecast_issue_calendar_month'].iloc[0]]
        fig, axs = plt.subplots(nrows=2, ncols=max(len(crops), 2), figsize=(14,10))  # need two at least for the loop below
        ax_c = 0  # ax counter
        # get max mteric
        ymax = np.max([b1[b1[var4time] == t][metric2use].max(),  b1[b1[var4time] == t]['rRMSE_p_areaWeighted'].max()])
        for crop in crops:
            # in order to assign the same colors I have to do some workaround
            tmp = b1[(b1[var4time] == t) & (b1['Crop'] == crop)].copy()
            # sort_dict = {'Null_model': 0, 'Trend': 1, 'PeakNDVI': 2, 'ML': 3}
            # Tab change 2025
            # sort_dict = {'Null_model': 0, 'Trend': 1, 'PeakNDVI': 2, 'ML': 3, 'BestByAdmin': 4}
            sort_dict = {'Null_model': 0, 'Trend': 1, 'PeakNDVI': 2, 'Tab': 3, 'ML': 4, 'BestByAdmin': 5}
            tmp['pltOrder'] = tmp['tmp_est'].map(sort_dict)
            tmp = tmp.sort_values('pltOrder')
            # get area weigthed rRMSE
            # tmp = areaWeighted_rRMSE(tmp, df_regNames, df_Stats)
            p = sns.barplot(tmp, x="tmp_est", y=metric2use, hue="tmp_est",
                            palette=colors, ax=axs[0, ax_c], dodge=False, width=0.4, legend="full")
            ml_row = tmp[tmp['tmp_est'] == 'ML']
            [info_string0, info_string1, info_string2, info_string3] = output_row_to_ML_info_string(ml_row, metric2use)
            axs[0, ax_c].text(x='ML', y=-0.1, s= metric2use + ' = ' + info_string0, ha='center', transform=axs[0, ax_c].get_xaxis_transform())
            axs[0, ax_c].text(x='ML', y=-0.14, s= info_string1, ha='center', transform=axs[0, ax_c].get_xaxis_transform())
            axs[0, ax_c].text(x='ML', y=-0.18, s= info_string2, ha='center', transform=axs[0, ax_c].get_xaxis_transform())
            axs[0, ax_c].text(x='ML', y=-0.22, s= info_string3, ha='center', transform=axs[0, ax_c].get_xaxis_transform())
            axs[0, ax_c].get_legend().set_visible(False)
            text = ''
            if bool(config.crop_au_exclusions):
                if crop in config.crop_au_exclusions.keys():
                    text = ', ML omitting: ' + ",".join(config.crop_au_exclusions[crop])
            axs[0, ax_c].set_title(crop + text)
            axs[0, ax_c].set(ylim=(0, ymax * 1.1))
            axs[0, ax_c].set(xlabel='')
            ax_c = ax_c + 1
        h, l = p.get_legend_handles_labels()
        axs[0, ax_c-1].legend(h, l, title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax_c = 0

        # Second row fo area weighted
        for crop in crops:
            # in order to assign the same colors I have to do some workaround
            tmp = b1[(b1[var4time] == t) & (b1['Crop'] == crop)].copy()
            # sort_dict = {'Null_model': 0, 'Trend': 1, 'PeakNDVI': 2, 'ML': 3}
            tmp['pltOrder'] = tmp['tmp_est'].map(sort_dict)
            tmp = tmp.sort_values('pltOrder')
            p = sns.barplot(tmp, x="tmp_est", y='rRMSE_p_areaWeighted', hue="tmp_est",
                            palette=colors, ax=axs[1, ax_c], dodge=False, width=0.4, legend="full")
            ml_row = tmp[tmp['tmp_est'] == 'ML']
            [info_string0, info_string1, info_string2, info_string3] = output_row_to_ML_info_string(ml_row,
                                                                                                    metric2use)
            info_string0 = str(round(ml_row['rRMSE_p_areaWeighted'].values[0], 2))
            axs[1, ax_c].text(x='ML', y=-0.1, s=metric2use + '_AW = ' + info_string0, ha='center',
                              transform=axs[1, ax_c].get_xaxis_transform())
            axs[1, ax_c].text(x='ML', y=-0.14, s=info_string1, ha='center',
                              transform=axs[1, ax_c].get_xaxis_transform())
            axs[1, ax_c].text(x='ML', y=-0.18, s=info_string2, ha='center',
                              transform=axs[1, ax_c].get_xaxis_transform())
            axs[1, ax_c].text(x='ML', y=-0.22, s=info_string3, ha='center',
                              transform=axs[1, ax_c].get_xaxis_transform())
            axs[1, ax_c].get_legend().set_visible(False)
            text = ''
            if bool(config.crop_au_exclusions):
                if crop in config.crop_au_exclusions.keys():
                    text = ', ML omitting: ' + ",".join(config.crop_au_exclusions[crop])
            axs[1, ax_c].set_title(crop + text)
            axs[1, ax_c].set(ylim=(0, ymax * 1.1))
            axs[1, ax_c].set(xlabel='')
            ax_c = ax_c + 1
        if len(crops) == 1:
            axs[0, 1].remove()
            axs[1, 1].remove()
        fig.text(0.90, 0.775, 'rRMSE', ha='left')

        fig.text(0.90, 0.275, 'Crop-area weighted', ha='left')
        fig.text(0.90, 0.255, 'rRMSE', ha='left')
        plt.tight_layout()
        forecastingPrct = config.forecastingPrct[config.forecastingMonths.index(t)]
        fig_name = outputDir + '/' + 'forecast_mInSeas' + str(t) + '_early_' + str(forecast_issue_calendar_month) + '_prctSeas' + str(forecastingPrct) + '_all_crops_performances.png'
        plt.savefig(fig_name)
        plt.close(fig)

def summary_stats(b1, config, var4time, outputDir):
    # compute a summary stats file for hindcasting

    # get overall stats (dropping AU duplicated, here I have for the same run ID results for each AU, here
    # I am interested only in overall results, that is the same for all duplicates)
    b1_overall = b1.drop_duplicates(subset='runID', keep='first').copy()
    df = pd.DataFrame(
        columns=['Crop', 'Prct_area_used', 'Prct_season_forecasts', 'Calendar_month_forecast', 'ML_omit_admins',
                 'Benchs_better_than_ML', 'ML_estimator', 'ML_rRMSEp', 'ML_R2_overall', 'BestByAdmin_rRMSEp'])
    for t in b1[var4time].unique():
        # get forecast % forecast_issue_calendar_month
        forecastingPrct = config.forecastingPrct[config.forecastingMonths.index(t)]
        forecast_issue_calendar_month = calendar.month_abbr[b1[b1[var4time] == t]['forecast_issue_calendar_month'].iloc[0]]
        crops = b1['Crop'].unique()

        # get % area anlysed
        for crop in crops:
            # print(t, crop)
            tmp = b1_overall[(b1_overall[var4time] == t) & (b1_overall['Crop'] == crop)].copy()
            txt_excl = 'none'
            if bool(config.crop_au_exclusions):
                if crop in config.crop_au_exclusions.keys():
                    txt_excl = config.crop_au_exclusions[crop]
            # benchmark better than models
            rRMSE_p_ML = tmp.loc[tmp['tmp_est'] == 'ML']['rRMSE_p'].iloc[0]
            Benchs_better_than_ML = tmp[(tmp['rRMSE_p'] < rRMSE_p_ML) & (tmp['Estimator'] != 'BestByAdmin')]['Estimator'].unique()
            if len(Benchs_better_than_ML) > 0:
                Benchs_better_than_ML = str(Benchs_better_than_ML)
            else:
                Benchs_better_than_ML = 'none'
            row = pd.DataFrame([[crop, config.prct2retain, forecastingPrct, forecast_issue_calendar_month, txt_excl, Benchs_better_than_ML, tmp[tmp['tmp_est'] == 'ML']['Estimator'].iloc[0],
                                tmp[tmp['tmp_est'] == 'ML']['rRMSE_p'].iloc[0],
                                tmp[tmp['tmp_est'] == 'ML']['avg_R2_p_overall'].iloc[0], tmp[tmp['tmp_est'] == 'BestByAdmin']['rRMSE_p'].iloc[0]]],
                               columns=['Crop', 'Prct_area_used', 'Prct_season_forecasts','Calendar_month_forecast','ML_omit_admins', 'Benchs_better_than_ML', 'ML_estimator', 'ML_rRMSEp', 'ML_R2_overall', 'BestByAdmin_rRMSEp'])
            df = pd.concat([df, row], ignore_index=True)

    best_times = df.loc[df.groupby('Crop')['BestByAdmin_rRMSEp'].idxmin()][['Crop', 'Prct_season_forecasts']].set_index(
        'Crop')
    df['BestTime'] = df['Crop'].map(best_times['Prct_season_forecasts'])
    df.insert(3, 'BestTime', df.pop('BestTime'))
    df.to_csv(outputDir + '/all_crops_performances.csv', index=False)


def scatter_plots_and_maps(b1, config, mlsettings, var4time, OutputDir, fn_shape_gaul1, country_name_in_shp_file,  gdf_gaul0_column='name0'): #onfig, fn_shape_gaul1, country_name_in_shp_file,  gdf_gaul0_column='name0'

    df_regNames = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_REGION_id.csv'))
    crops = b1['Crop'].unique()
    forcTimes = b1[var4time].unique()
    if isinstance(config.fn_reference_shape, list): #It is a multi-shp like Morocco
        fp = b1['last_shp'].iloc[0]
    else:
        fp = fn_shape_gaul1
    gdf = gpd.read_file(fp)
    gdf_gaul1_id = config.adminID_column_name_in_shp_file
    for c in crops:
        for t in forcTimes:
            # get forecast_issue_calendar_month
            forecast_issue_calendar_month = calendar.month_abbr[b1[b1[var4time] == t]['forecast_issue_calendar_month'].iloc[0]]
            # get forecastingPrct from t (forecastingMonths)
            forecastingPrct = config.forecastingPrct[config.forecastingMonths.index(t)]
            # Tab change 2025
            # fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
            # fig is scatter by admin
            fig, axs = plt.subplots(3, 2, figsize=(15, 10), constrained_layout=True)
            axs = axs.flatten()
            # Tab change 2025
            # fig2, axs2 = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
            # fig2 is scatter by year
            fig2, axs2 = plt.subplots(3, 2, figsize=(15, 10), constrained_layout=True)
            axs2 = axs2.flatten()
            df_c_t = b1[(b1['Crop'] == c) & (b1[var4time] == t)].copy()
            # sort_dict = {'Null_model': 0, 'Trend': 1, 'PeakNDVI': 2, 'ML': 3}
            sort_dict = {'Null_model': 0, 'Trend': 1, 'PeakNDVI': 2, 'Tab': 3, 'ML': 4}
            df_c_t['pltOrder'] = df_c_t['tmp_est'].map(sort_dict)
            df_c_t = df_c_t.sort_values('pltOrder').reset_index()
            # ax holder for rrmse by AU
            # fig3 is the map of error
            fig3, axs3 = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
            # reorder conveniently
            axs3 = [axs3[0, 0], axs3[0, 1], axs3[1, 0], axs3[1, 1], axs3[0, 2], axs3[1, 2]]
            minOfMins = df_c_t.rrmse_prct.min()
            maxOfmaxs = df_c_t.rrmse_prct.max()
            dfBestPerAU = df_c_t[df_c_t['rrmse_prct'] == df_c_t.groupby(['adm_id'])['rrmse_prct'].transform(min)]
            # colors = {'ML': "#0000FF", 'Null_model': "#969696", 'PeakNDVI': "#FF0000", 'Trend': "#009600"}
            colors = {'ML': "#0000FF", 'Null_model': "#969696", 'PeakNDVI': "#FF0000", 'Trend': "#009600", 'Tab': "#660066"}
            dfBestPerAU['colors'] = dfBestPerAU['tmp_est'].map(colors)
            index = 0
            for model in df_c_t['tmp_est'].unique():
                df_c_t_m = df_c_t[df_c_t['tmp_est'] == model]
                e50_yield_data_analysis.mapDfColumn2Ax(df_c_t_m, 'adm_id', 'rrmse_prct', 'adm_name',
                                                                     gdf, gdf_gaul1_id, gdf_gaul0_column,
                                                                     country_name_in_shp_file,
                                                                     'rRMSE (%)', cmap='tab20b',
                                                                     # minmax=[minOfMins, maxOfmaxs], ax=axs3[index])
                                                                     ax=axs3[index])
                axs3[index].set_title(model)
                # get mres errors
                runID = df_c_t_m['runID'].iloc[0]
                est = df_c_t_m['Estimator'].iloc[0]
                myID = f'{runID:06d}'
                fn_spec = os.path.join(pathlib.Path(config.models_spec_dir), myID + '_' + c + '_' + est + '.json')
                # print(fn_spec)
                df = d090_model_wrapper.fit_and_validate_single_model(fn_spec, config, 'tuning', run2get_mres_only=True)
                lims = [np.floor(np.nanmin([df['yLoo_true'].values, df['yLoo_pred'].values])),
                        np.ceil(np.nanmax([df['yLoo_true'].values, df['yLoo_pred'].values]))]
                r2p = d140_modelStats.r2_nan(df['yLoo_true'].values, df['yLoo_pred'].values)
                for au_code in df['adm_id'].unique():
                    x = df[df['adm_id'] == au_code]['yLoo_true'].values
                    y = df[df['adm_id'] == au_code]['yLoo_pred'].values
                    lbl = df_regNames[df_regNames['adm_id'] == au_code.astype('int')]['adm_name'].values[0]
                    axs[index].scatter(x, y, label=lbl, edgecolor='k', linewidth=0.5)
                    axs[index].plot(lims, lims, color='black', linewidth=0.5)
                    axs[index].set_title(est + ',R2p=' + str(np.round(r2p, 2)))
                    axs[index].set_xlim(lims)
                    axs[index].set_ylim(lims)
                    axs[index].set_xlabel('Obs')
                    axs[index].set_ylabel('Pred')
                    axs[index].legend(frameon=False, loc='upper left')
                color = iter(cm.gist_ncar(np.linspace(0, 1, len(df['Year'].unique())))) # rainbow too small, gist_rainbow
                for yr in df['Year'].unique():
                    clr = next(color)
                    x = df[df['Year'] == yr]['yLoo_true'].values
                    y = df[df['Year'] == yr]['yLoo_pred'].values
                    lbl = str(int(yr))
                    axs2[index].scatter(x, y, label=lbl, c=clr, edgecolor='k', linewidth=0.5)
                    axs2[index].plot(lims, lims, color='black', linewidth=0.5)
                    axs2[index].set_title(est + ',R2p=' + str(np.round(r2p, 2)))
                    axs2[index].set_xlim(lims)
                    axs2[index].set_ylim(lims)
                    axs2[index].set_xlabel('Obs')
                    axs2[index].set_ylabel('Pred')
                    axs2[index].legend(frameon=False, ncol=4, loc='upper left', prop={'size':10}, handletextpad=0.005, columnspacing=0.02, labelspacing=0.05)
                index = index + 1
            fig.tight_layout()
            fig2.tight_layout()
            fig_name = OutputDir + '/' + 'forecast_mInSeas' + str(t) + '_early_' + str(
                forecast_issue_calendar_month) + '_prctSeas' + str(forecastingPrct) + '-' + c + '_scatter_by_admin.png'
            fig.savefig(fig_name)
            fig_name = OutputDir + '/' + 'forecast_mInSeas' + str(t) + '_early_' + str(
                forecast_issue_calendar_month) + '_prctSeas' + str(forecastingPrct) + '-' + c + '_scatter_by_year.png'
            fig2.savefig(fig_name)
            # Tab change 2025
            # now plot the best by au (ax 5 instead of 4 with Tab)
            e50_yield_data_analysis.mapDfColumn2Ax(dfBestPerAU, 'adm_id', 'Estimator', 'adm_name', gdf,
                                                             gdf_gaul1_id, gdf_gaul0_column,
                                                             country_name_in_shp_file,
                                                             'Estimator', ax=axs3[5], cate=True)
            axs3[5].set_title('Best')
            if 'Tab' not in df_c_t['tmp_est'].unique():
                axs3[4].set_axis_off()
            # axs3[5].set_axis_off()
            fig3.subplots_adjust(wspace=0.01, hspace=0.01)
            for ax in axs3:
                ax.set_anchor('NW')
                # for child in fig3.get_children():
                #     if hasattr(child, 'set_bbox_to_anchor'):
                #         child.set_bbox_to_anchor((0.5, -0.2))

            fig_name = OutputDir + '/' + 'forecast_mInSeas' + str(t) + '_early_' + str(
                forecast_issue_calendar_month) + '_prctSeas' + str(forecastingPrct) + '-' + c + '_AU_rrmse.png'
            fig3.savefig(fig_name)
            plt.close(fig)
            plt.close(fig2)
            plt.close(fig3)




def accuracy_over_time(mRes, mCountryRes, filename=None):
    # Define dash style
    # there are many AUs, add a columnn to be used with style, 1,2,3,4,5,1,2,3,4,5,1,2..
    # suppress graphics for massive runs
    hideGraph = True
    if (hideGraph):
        plt.ioff()
        plt.switch_backend('agg')  # attempt to get rid of multi thread erros,
        # to have the graph displayed: # plt.ion() # plt.switch_backend('TkAgg') #plt.show()

    dash_base_styles = ["",
                        (4, 1.5),
                        (1, 1),
                        (3, 1, 1, 1, 1, 1),
                        (5, 1, 1, 1)]
    dash_styles = dash_base_styles * 100
    # dash_styles = dash_styles[0:len(mRes)]
    dash_styles = dash_styles[0:len(mRes['adm_id'].unique())]

    # line plot
    fig, axs = plt.subplots(nrows=2, figsize=(12, 6))
    minV, maxV = mRes[['yLoo_true', 'yLoo_pred']].values.min(), mRes[
        ['yLoo_true', 'yLoo_pred']].values.max()
    margin = (maxV - minV) / 20.0
    g = sns.lineplot(x="Year", y="yLoo_true", hue="adm_id", style="adm_id", data=mRes, ax=axs[0], dashes=dash_styles,
                     palette="Spectral", legend='full')
    axs[0].plot(mCountryRes.index, mCountryRes['yLoo_true'], color='green', linewidth=3, linestyle='dashed')
    axs[0].set_ylim(minV - margin, maxV + margin)
    axs[0].axes.set_xlabel('')
    # resize to accomodat legend
    box = g.get_position()
    g.set_position([box.x0, box.y0, box.width * 0.9, box.height])  # resize position
    # Put a legend to the right side
    g.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)

    g = sns.lineplot(x="Year", y="yLoo_pred", hue="adm_id", style="adm_id", data=mRes, ax=axs[1], dashes=dash_styles,
                     palette="Spectral", legend='full')
    axs[1].plot(mCountryRes.index, mCountryRes['yLoo_pred'], color='green', linewidth=3, linestyle='dashed')
    axs[1].set_ylim(minV - margin, maxV + margin)
    # resize to accomodat legend
    box = g.get_position()
    g.set_position([box.x0, box.y0, box.width * 0.9, box.height])  # resize position
    g.legend()._visible = False

    if filename is not None:
        fig.savefig(filename.replace(" ", ""))
    plt.close(fig)

    return

def scatter_plot_accuracy(mRes, title_R2, color_factor, filename=None):
    """
    Scatter plot coloring by a factor
    """
    # suppress graphics for massive runs
    hideGraph = True
    if (hideGraph):
        plt.ioff()
        plt.switch_backend('agg')  # attempt to get rid of multi thread erros,
        # to have the graph displayed: # plt.ion() # plt.switch_backend('TkAgg') #plt.show()

    dash_base_styles = ["",
                        (4, 1.5),
                        (1, 1),
                        (3, 1, 1, 1, 1, 1),
                        (5, 1, 1, 1)]
    dash_styles = dash_base_styles * 100
    # dash_styles = dash_styles[0:len(mRes)]
    dash_styles = dash_styles[0:len(mRes['adm_id'].unique())]

    #scatter plot
    fig = plt.figure(figsize = (6,6))
    base_markers = ('o', 'v', 's', '^', 'X')
    markers = base_markers * 100
    #markers = markers[0:len(mRes)]
    markers = markers[0:len(mRes[color_factor].unique())]
    g = sns.scatterplot(x="yLoo_true", y="yLoo_pred", hue=color_factor, style=color_factor, data=mRes, palette = "Spectral", markers=markers, legend='full')
    sns.regplot(x="yLoo_true", y="yLoo_pred", data=mRes, color='k', scatter_kws={"alpha": 0.0})
    #sns.jointplot(x="yLoo_true", y="yLoo_pred", hue="adm_id", data=mRes)
    minV, maxV = mRes[['yLoo_true', 'yLoo_pred']].values.min(), mRes[['yLoo_true', 'yLoo_pred']].values.max()
    margin = (maxV-minV)/20.0
    plt.ylim(minV-margin, maxV+margin)
    plt.xlim(minV-margin, maxV+margin)
    plt.plot([minV-margin, maxV+margin], [minV-margin, maxV+margin], color= 'k',linewidth=1, linestyle='dashed')
    plt.title('R2_pred='+str(np.round(title_R2,2)))
    box = g.get_position()
    g.set_position([box.x0, box.y0, box.width * 0.85, box.height* 0.85])
    g.legend(loc='lower left', bbox_to_anchor=(1, 0), ncol=1)

    if filename is not None:
        fig.savefig(filename.replace(" ", ""))
    plt.close(fig)


