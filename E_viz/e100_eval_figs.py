import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import pathlib
import calendar
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



def bars_by_forecast_time2(b1, config, metric2use, mlsettings, var4time, outputDir):
    # In this version I compute the rel RMSE by admin (aeach admin with its own mean yield) and then I weight them
    # using area of the last five years (to occount for the fact that larger errors are more tolerable if the area is small)
    os.path.join(config.data_dir, 'Label_analysis')
    df_Stats5yrs = pd.read_csv(os.path.join(os.path.join(config.data_dir, 'Label_analysis'+str(config.prct2retain)), config.AOI + '_5yrsStats_retainPRCT100.csv'))
    # Now read mres, compute metric2use at the admin level, and make an area average add as a new columns
    b1['rRMSE_p_areaWeighted'] = -999
    b1 = b1.reset_index() # the index was repeating
    dfAU = pd.DataFrame()
    for index, row in b1.iterrows():
        # get run_id
        runID = row['runID']
        #est = row['Estimator']
        myID = f'{runID:06d}'
        fn_mRes_out = os.path.join(config.models_out_dir, 'ID_' + str(myID) +
                                   '_crop_' + row['Crop'] + '_Yield_' + row['Estimator'] +
                                   '_mres.csv')
        mRes = pd.read_csv(fn_mRes_out)
        rRMSE_pByAdmin = d140_modelStats.statsByAdmin(mRes)
        rRMSE_pByAdmin = rRMSE_pByAdmin.merge(df_Stats5yrs[df_Stats5yrs['Crop_name|first']==row['Crop']], how='left', left_on='adm_id', right_on='adm_id|')
        b1.at[index, 'rRMSE_p_areaWeighted'] = np.average(rRMSE_pByAdmin.rrmse_prct, weights=rRMSE_pByAdmin['Area|mean'])
        rRMSE_pByAdmin.insert(1, column='Estimator', value=row['Estimator'])
        rRMSE_pByAdmin.insert(2, column='forecast_time', value=row['forecast_time'])

        dfAU = pd.concat([dfAU, rRMSE_pByAdmin])
    dfAU.to_csv(os.path.join(outputDir, 'all_model_best1_AU_error.csv'))


    # in order to assign the same colors and keep a defined order I have to do some workaround
    b1['tmp_est'] = b1['Estimator'].map(lambda x: x if x in mlsettings.benchmarks else 'ML')
    # colors = {'Cat1': "#F28E2B", 'Cat2': "#4E79A7", 'Cat3': "#79706E"}
    colors = {'ML': "#0000FF", 'Null_model': "#969696", 'PeakNDVI': "#FF0000", 'Trend': "#009600"}
    for t in b1[var4time].unique():
        crops = b1['Crop'].unique()
        # get forecast_issue_calendar_month
        forecast_issue_calendar_month = calendar.month_abbr[
            b1[b1[var4time] == t]['forecast_issue_calendar_month'].iloc[0]]
        fig, axs = plt.subplots(nrows=2, ncols=max(len(crops), 2), figsize=(14,10))  # need two at least for the loop below
        # fig, axs = plt.subplots(nrows=2, ncols=len(crops), figsize=(14, 12))
        ax_c = 0  # ax counter
        # get max metirc
        ymax = b1[b1[var4time] == t][metric2use].max()
        for crop in crops:
            # in order to assign the same colors I have to do some workaround
            tmp = b1[(b1[var4time] == t) & (b1['Crop'] == crop)].copy()

            sort_dict = {'Null_model': 0, 'Trend': 1, 'PeakNDVI': 2, 'ML': 3}
            tmp['pltOrder'] = tmp['tmp_est'].map(sort_dict)
            tmp = tmp.sort_values('pltOrder')
            # get area weigthed rRMSE
            # tmp = areaWeighted_rRMSE(tmp, df_regNames, df_Stats)
            p = sns.barplot(tmp, x="tmp_est", y=metric2use, hue="tmp_est",
                            palette=colors, ax=axs[0, ax_c], dodge=False, width=0.4, legend="full")
            ml_row = tmp[tmp['tmp_est'] == 'ML']
            [info_string0, info_string1, info_string2, info_string3] = output_row_to_ML_info_string(ml_row, metric2use)
            axs[0, ax_c].text(0.875, -0.1, metric2use + ' = ' + info_string0, transform=axs[0, ax_c].transAxes, horizontalalignment='center')
            axs[0, ax_c].text(0.875, -0.14, info_string1, transform=axs[0, ax_c].transAxes, horizontalalignment='center')
            axs[0, ax_c].text(0.875, -0.18, info_string2, transform=axs[0, ax_c].transAxes, horizontalalignment='center')
            axs[0, ax_c].text(0.875, -0.22, info_string3, transform=axs[0, ax_c].transAxes, horizontalalignment='center')
            axs[0, ax_c].get_legend().set_visible(False)
            axs[0, ax_c].set_title(crop)
            axs[0, ax_c].set(ylim=(0, ymax * 1.1))
            axs[0, ax_c].set(xlabel='Model')
            ax_c = ax_c + 1
        h, l = p.get_legend_handles_labels()
        # h, l = axs[0, ax_c-1].get_legend_handles_labels()
        # plt.legend(h, l, title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        axs[0, ax_c-1].legend(h, l, title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax_c = 0
        # get max metirc
        ymax = b1[b1[var4time] == t]['rRMSE_p_areaWeighted'].max()
        for crop in crops:
            # in order to assign the same colors I have to do some workaround
            tmp = b1[(b1[var4time] == t) & (b1['Crop'] == crop)].copy()
            sort_dict = {'Null_model': 0, 'Trend': 1, 'PeakNDVI': 2, 'ML': 3}
            tmp['pltOrder'] = tmp['tmp_est'].map(sort_dict)
            tmp = tmp.sort_values('pltOrder')
            # get area weigthed rRMSE
            # tmp = areaWeighted_rRMSE(tmp, df_regNames, df_Stats)
            p = sns.barplot(tmp, x="tmp_est", y='rRMSE_p_areaWeighted', hue="tmp_est",
                            palette=colors, ax=axs[1, ax_c], dodge=False, width=0.4, legend="full")
            ml_row = tmp[tmp['tmp_est'] == 'ML']
            [info_string0, info_string1, info_string2, info_string3] = output_row_to_ML_info_string(ml_row,
                                                                                                    metric2use)
            info_string0 = str(round(ml_row['rRMSE_p_areaWeighted'].values[0], 2))
            axs[1, ax_c].text(0.875, -0.1, 'rRMSE_p_AW' + ' = ' + info_string0, transform=axs[1, ax_c].transAxes,
                              horizontalalignment='center')
            axs[1, ax_c].text(0.875, -0.14, info_string1, transform=axs[1, ax_c].transAxes,
                              horizontalalignment='center')
            axs[1, ax_c].text(0.875, -0.18, info_string2, transform=axs[1, ax_c].transAxes,
                              horizontalalignment='center')
            axs[1, ax_c].text(0.875, -0.22, info_string3, transform=axs[1, ax_c].transAxes,
                              horizontalalignment='center')
            axs[1, ax_c].get_legend().set_visible(False)
            axs[1, ax_c].set_title(crop)
            axs[1, ax_c].set(ylim=(0, ymax * 1.1))
            axs[1, ax_c].set(xlabel='Model')
            ax_c = ax_c + 1
        if len(crops) == 1:
            axs[0, 1].remove()
            axs[1, 1].remove()
        # h, l = p.get_legend_handles_labels()
        # plt.legend(h, l, title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        fig.tight_layout()
        #get forecastingPrct from t (forecastingMonths)
        forecastingPrct = config.forecastingPrct[config.forecastingMonths.index(t)]
        fig_name = outputDir + '/' + 'forecast_mInSeas' + str(t) + '_issue_early_' + str(forecast_issue_calendar_month) + '_prctSeas' + str(forecastingPrct) + '_all_crops_performances.png'
        plt.savefig(fig_name)
        plt.close(fig)

def _bars_by_forecast_time(b1, metric2use, mlsettings, var4time, outputDir):
    # in order to assign the same colors and keep a defined order I have to do some workaround
    b1['tmp_est'] = b1['Estimator'].map(lambda x: x if x in mlsettings.benchmarks else 'ML')
    # colors = {'Cat1': "#F28E2B", 'Cat2': "#4E79A7", 'Cat3': "#79706E"}
    colors = {'ML': "#0000FF", 'Null_model': "#969696", 'PeakNDVI': "#FF0000", 'Trend': "#009600"}
    for t in b1[var4time].unique():
        crops = b1['Crop'].unique()
        # get forecast_issue_calendar_month
        forecast_issue_calendar_month = calendar.month_abbr[b1[b1[var4time] == t]['forecast_issue_calendar_month'].iloc[0]]
        fig, axs = plt.subplots(ncols=max(len(crops),2), figsize=(14, 6)) #need two at least for the loop below
        ax_c = 0  # ax counter
        # get mas metirc
        ymax = b1[b1[var4time] == t][metric2use].max()
        for crop in crops:
            # in order to assign teh same colors I have to do some workaround
            tmp = b1[(b1[var4time] == t) & (b1['Crop'] == crop)].copy()
            sort_dict = {'Null_model': 0, 'Trend': 1, 'PeakNDVI': 2, 'ML': 3}
            tmp['pltOrder'] = tmp['tmp_est'].map(sort_dict)
            tmp = tmp.sort_values('pltOrder')
            p = sns.barplot(tmp, x="tmp_est", y=metric2use, hue="tmp_est",
                            palette=colors, ax=axs[ax_c], dodge=False, width=0.4, legend="full")
            ml_row = tmp[tmp['tmp_est'] == 'ML']
            [info_string0, info_string1, info_string2, info_string3] = output_row_to_ML_info_string(ml_row, metric2use)
            axs[ax_c].text(0.875, -0.1, metric2use + '= ' + info_string0, transform=axs[ax_c].transAxes,
                           horizontalalignment='center')
            axs[ax_c].text(0.875, -0.14, info_string1, transform=axs[ax_c].transAxes, horizontalalignment='center')
            axs[ax_c].text(0.875, -0.18, info_string2, transform=axs[ax_c].transAxes, horizontalalignment='center')
            axs[ax_c].text(0.875, -0.22, info_string3, transform=axs[ax_c].transAxes, horizontalalignment='center')
            axs[ax_c].get_legend().set_visible(False)
            axs[ax_c].set_title(crop)
            axs[ax_c].set(ylim=(0, ymax * 1.1))
            axs[ax_c].set(xlabel='Model')
            ax_c = ax_c + 1

        if len(crops) == 1:
            axs[1].remove()
        h, l = p.get_legend_handles_labels()
        plt.legend(h, l, title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        fig.tight_layout()

        plt.savefig(outputDir + '/' + 'all_model_best1_forecast_month_season_' + str(t) + '_issue_early_' + str(forecast_issue_calendar_month) + '.png')
        plt.close(fig)

def scatter_plots_and_maps(b1, config, mlsettings, var4time, OutputDir, fn_shape_gaul1, country_name_in_shp_file,  gdf_gaul0_column='name0'): #onfig, fn_shape_gaul1, country_name_in_shp_file,  gdf_gaul0_column='name0'
    # in order to assign the same colors and keep a defined order I have to do some workaround
    b1['tmp_est'] = b1['Estimator'].map(lambda x: x if x in mlsettings.benchmarks else 'ML')
    df_regNames = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_REGION_id.csv'))
    crops = b1['Crop'].unique()
    forcTimes = b1[var4time].unique()
    fp = fn_shape_gaul1
    gdf = gpd.read_file(fp)
    gdf_gaul1_id = "asap1_id"
    for c in crops:
        for t in forcTimes:
            # get forecast_issue_calendar_month
            forecast_issue_calendar_month = calendar.month_abbr[b1[b1[var4time] == t]['forecast_issue_calendar_month'].iloc[0]]
            # get forecastingPrct from t (forecastingMonths)
            forecastingPrct = config.forecastingPrct[config.forecastingMonths.index(t)]
            fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
            axs = axs.flatten()
            fig2, axs2 = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
            axs2 = axs2.flatten()
            df_c_t = b1[(b1['Crop'] == c) & (b1[var4time] == t)].copy()
            sort_dict = {'Null_model': 0, 'Trend': 1, 'PeakNDVI': 2, 'ML': 3}
            df_c_t['pltOrder'] = df_c_t['tmp_est'].map(sort_dict)
            df_c_t = df_c_t.sort_values('pltOrder').reset_index()
            # ax holder for rrmse by AU
            fig3, axs3 = plt.subplots(2, 3, figsize=(12, 10), constrained_layout=True)
            # reorder coneniently
            axs3 = [axs3[0, 0], axs3[0, 1], axs3[1, 0], axs3[1, 1], axs3[0, 2], axs3[1, 2]]


            # iterate one time to get min max over pandas df
            dfAU = pd.DataFrame()
            for index, row in df_c_t.iterrows():
                # get run_id
                runID = row['runID']
                est = row['Estimator']
                tmp_est = row['tmp_est']
                myID = f'{runID:06d}'
                fn_spec = os.path.join(pathlib.Path(config.models_spec_dir), myID + '_' + c + '_' + est + '.json')
                print(fn_spec)
                df = d090_model_wrapper.fit_and_validate_single_model(fn_spec, config, 'tuning', run2get_mres_only=True)
                statsByAdmin = d140_modelStats.statsByAdmin(df)
                statsByAdmin = statsByAdmin.merge(df_regNames, how='left', left_on='adm_id', right_on='adm_id')
                statsByAdmin['Estimator'] = tmp_est
                dfAU = pd.concat([dfAU, statsByAdmin])
            minOfMins = dfAU.rrmse_prct.min()
            maxOfmaxs = dfAU.rrmse_prct.max()
            # get best by admin
            dfBestPerAU = dfAU[dfAU['rrmse_prct'] == dfAU.groupby(['adm_id'])['rrmse_prct'].transform(min)]
            colors = {'ML': "#0000FF", 'Null_model': "#969696", 'PeakNDVI': "#FF0000", 'Trend': "#009600"}
            dfBestPerAU['colors'] = dfBestPerAU['Estimator'].map(colors)
            # iterate over pandas df
            for index, row in df_c_t.iterrows():
                # get run_id
                runID = row['runID']
                est = row['Estimator']
                tmp_est= row['tmp_est']
                myID = f'{runID:06d}'
                fn_spec = os.path.join(pathlib.Path(config.models_spec_dir), myID + '_' + c + '_' + est + '.json')
                print(fn_spec)
                df = d090_model_wrapper.fit_and_validate_single_model(fn_spec, config, 'tuning', run2get_mres_only=True)
                statsByAdmin = d140_modelStats.statsByAdmin(df)
                statsByAdmin = statsByAdmin.merge(df_regNames, how='left', left_on='adm_id', right_on='adm_id')
                axs3[index] = e50_yield_data_analysis.mapDfColumn2Ax(statsByAdmin, 'adm_id', 'rrmse_prct', 'adm_name', gdf, gdf_gaul1_id, gdf_gaul0_column, country_name_in_shp_file,
                                'rRMSE (%)', cmap='tab20b', minmax=[minOfMins, maxOfmaxs], ax=axs3[index])
                axs3[index].set_title(tmp_est)
                # fig_name = OutputDir + '/' + 'forecast_mInSeas' + str(t) + '_issue_early_' + str(forecast_issue_calendar_month) + '_prctSeas' +  str(forecastingPrct)+ '-' + c + '_' + tmp_est +'_AU_rrmse.png'
                # e50_yield_data_analysis.mapDfColumn(statsByAdmin, 'adm_id', 'rrmse_prct', 'adm_name', gdf, gdf_gaul1_id, gdf_gaul0_column, country_name_in_shp_file,
                # 'rRMSE (%)', cmap='tab20b', minmax=None, fn_fig=fig_name, ax=None)
                lims = [np.floor(np.min([df['yLoo_true'].values, df['yLoo_pred'].values])),
                        np.ceil(np.max([df['yLoo_true'].values, df['yLoo_pred'].values]))]
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
            fig.tight_layout()
            fig2.tight_layout()
            fig_name = OutputDir + '/' + 'forecast_mInSeas' + str(t) + '_issue_early_' + str(forecast_issue_calendar_month) + '_prctSeas' + str(forecastingPrct) + '-' + c + '_scatter_by_admin.png'
            fig.savefig(fig_name)
            fig_name = OutputDir + '/' + 'forecast_mInSeas' + str(t) + '_issue_early_' + str(forecast_issue_calendar_month) + '_prctSeas' + str(forecastingPrct) + '-' + c + '_scatter_by_year.png'
            fig2.savefig(fig_name)

            # now plot the best by au
            axs3[4] = e50_yield_data_analysis.mapDfColumn2Ax(dfBestPerAU, 'adm_id', 'tmp_est', 'adm_name', gdf,
                                                                 gdf_gaul1_id, gdf_gaul0_column,
                                                                 country_name_in_shp_file,
                                                                 'Estimator', ax=axs3[4], cate=True)
            axs3[4].set_title('Best')

            axs3[5].set_axis_off()
            fig3.subplots_adjust(wspace=0.01, hspace=0.01)
            for ax in axs3:
                ax.set_anchor('NW')
            fig_name = fig_name = OutputDir + '/' + 'forecast_mInSeas' + str(t) + '_issue_early_' + str(forecast_issue_calendar_month) + '_prctSeas' +  str(forecastingPrct)+ '-' + c  +'_AU_rrmse.png'
            # fig3.tight_layout()
            fig3.savefig(fig_name)

            plt.close(fig)
            plt.close(fig2)
            plt.close(fig3)

# def scatter_plots_and_maps(b1, config, var4time, OutputDir, fn_shape_gaul1, country_name_in_shp_file,  gdf_gaul0_column='name0'): #onfig, fn_shape_gaul1, country_name_in_shp_file,  gdf_gaul0_column='name0'
#     df_regNames = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_REGION_id.csv'))
#     crops = b1['Crop'].unique()
#     forcTimes = b1[var4time].unique()
#     fp = fn_shape_gaul1
#     gdf = gpd.read_file(fp)
#     gdf_gaul1_id = "asap1_id"
#     for c in crops:
#         for t in forcTimes:
#             fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
#             axs = axs.flatten()
#             df_c_t = b1[(b1['Crop'] == c) & (b1[var4time] == t)].copy()
#             sort_dict = {'Null_model': 0, 'Trend': 1, 'PeakNDVI': 2, 'ML': 3}
#             df_c_t['pltOrder'] = df_c_t['tmp_est'].map(sort_dict)
#             df_c_t = df_c_t.sort_values('pltOrder').reset_index()
#             # iterate over pandas df
#             for index, row in df_c_t.iterrows():
#                 # get run_id
#                 runID = row['runID']
#                 est = row['Estimator']
#                 myID = f'{runID:06d}'
#                 fn_spec = os.path.join(pathlib.Path(config.models_spec_dir), myID + '_' + c + '_' + est + '.json')
#                 print(fn_spec)
#                 df = d090_model_wrapper.fit_and_validate_single_model(fn_spec, config, 'tuning' , run2get_mres_only=True)
#                 statsByAdmin = d140_modelStats.statsByAdmin(df)
#                 statsByAdmin = statsByAdmin.merge(df_regNames, how='left', left_on='adm_id', right_on='adm_id')
#                 fig_name = OutputDir + '/' + 'all_model_best1_forecast_time_' + str(t) + '_' + c +'_AU_rrmse.png'
#                 e50_yield_data_analysis.mapDfColumn(statsByAdmin, 'adm_id', 'rrmse_prct', 'adm_name', gdf, gdf_gaul1_id, gdf_gaul0_column, country_name_in_shp_file,
#                 'rRMSE (%)', cmap='tab20b', minmax=None, fn_fig=fig_name, ax=None)
#                 lims = [np.floor(np.min([df['yLoo_true'].values, df['yLoo_pred'].values])),
#                         np.ceil(np.max([df['yLoo_true'].values, df['yLoo_pred'].values]))]
#                 r2p = d140_modelStats.r2_nan(df['yLoo_true'].values, df['yLoo_pred'].values)
#                 for au_code in df['adm_id'].unique():
#                     x = df[df['adm_id'] == au_code]['yLoo_true'].values
#                     y = df[df['adm_id'] == au_code]['yLoo_pred'].values
#                     lbl = df_regNames[df_regNames['adm_id'] == au_code.astype('int')]['adm_name'].values[0]
#                     axs[index].scatter(x, y, label=lbl)
#                     axs[index].plot(lims, lims, color='black', linewidth=0.5)
#                     axs[index].set_title(est + ',R2p=' + str(np.round(r2p, 2)))
#                     axs[index].set_xlim(lims)
#                     axs[index].set_ylim(lims)
#                     axs[index].set_xlabel('Obs')
#                     axs[index].set_ylabel('Pred')
#                     axs[index].legend(frameon=False, loc='upper left')
#             fig.tight_layout()
#             plt.savefig(OutputDir + '/' + 'all_model_best1_forecast_time_' + str(t) + '_' + c +'_scatter.png')
#             plt.close(fig)


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


