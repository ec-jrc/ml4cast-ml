import pandas as pd
import numpy as np
import os
import copy
import re
from B_preprocess import b101_load_cleaned
from E_viz import e110_ope_figs
import datetime
import glob
from pathlib import Path





def percentile_below(x, xt):
    """
    Retrieves the percentile at which a target value is observed
    :param x: distribution
    :param xt: target value
    :return: percentile value at which xt is observed
    """
    if xt > x.max():
        return 1
    p = np.arange(0, 101, 1)
    v = np.percentile(x, p)
    idx = np.where(xt < v)[0][0]
    return p[idx]/100

# def to_csv(self, regions, forecasts, funcertainty, fmae, runID = ''):
# def to_csv(config, forecast_issue_calendar_month, uset, regions, forecasts, rMAE_p_hindcasting, runID=''):
def to_csv(config, forecast_issue_calendar_month, uset, regions, forecasts, runID=''):
    df_forecast = pd.DataFrame({'adm_id': regions,
                                'Region_name': np.nan,
                                'Crop_name': uset['crop'],
                                'fyield': forecasts,
                                # 'fyield_SD_Bootstrap_1yr': funcertainty,
                                'fyield_rRMSEp_prct_hindcasting': np.nan,
                                # 'fyield_rMAEp_hindcasting': rMAE_p_hindcasting,
                                'fyield_percentile': np.nan,
                                'avg_obs_yield': np.nan,
                                'avg_obs_yield_last5yrs': np.nan,
                                'min_obs_yield': np.nan,
                                'max_obs_yield': np.nan,
                                '10percentile_obs_yield': np.nan,
                                '90percentile_obs_yield': np.nan,
                                'fyield_diff_pct (last 5 yrs in data avail)': np.nan,
                                'avg_obs_area_last5yrs': np.nan,
                                'fproduction(fyield*avg_obs_area_last5yrs)': np.nan,
                                'fproduction_percentile': np.nan,
                                'algorithm': uset['algorithm'],
                                'runID': runID})

    # get yield stats
    stats = b101_load_cleaned.LoadCleanedLabel(config)
    stats = stats[stats['Crop_name'] == uset['crop']]

    #get error by au
    defAuError =  pd.read_csv(os.path.join(config.models_out_dir, 'Analysis', 'all_model_best1_AU_error.csv'))
    defAuError = defAuError[(defAuError['forecast_time'] == uset['forecast_time']) & (defAuError['Crop_name|first'] == uset['crop']) & (defAuError['Estimator'] == uset['algorithm'])]
    for region in regions:
        # get stats for region and sort by year (*)to take last 5)
        stats_region = stats[stats['adm_id'] == region].sort_values(by=['Year'])
        defAuError_region = defAuError[defAuError['adm_id'] == region]
        fyield_region = df_forecast.loc[df_forecast['adm_id'] == region, 'fyield'].values
        # try:
        df_forecast.loc[df_forecast['adm_id'] == region, 'fyield_rRMSEp_prct_hindcasting'] = defAuError_region.rrmse_prct.values[0]
        # except:
        #     print('debug tru f110 line 67')
        df_forecast.loc[df_forecast['adm_id'] == region, 'adm_id'] = stats_region.iloc[0]['adm_id']
        df_forecast.loc[df_forecast['adm_id'] == region, 'fyield_percentile'] = \
            percentile_below(stats_region['Yield'], fyield_region)
        df_forecast.loc[df_forecast['adm_id'] == region, 'Region_name'] = \
            stats_region.iloc[0]['adm_name']
        df_forecast.loc[df_forecast['adm_id'] == region, 'fproduction(fyield*avg_obs_area_last5yrs)'] = \
            df_forecast.loc[df_forecast['adm_id'] == region, 'fyield'] * stats_region['Area'][-5::].mean()
        df_forecast.loc[df_forecast['adm_id'] == region, 'avg_obs_yield'] = stats_region['Yield'].mean()
        df_forecast.loc[df_forecast['adm_id'] == region, 'avg_obs_yield_last5yrs'] = stats_region['Yield'][-5::].mean() #avg_obs_yield_last5yrs
        df_forecast.loc[df_forecast['adm_id'] == region, 'min_obs_yield'] = stats_region['Yield'].min()
        df_forecast.loc[df_forecast['adm_id'] == region, 'max_obs_yield'] = stats_region['Yield'].max()
        df_forecast.loc[df_forecast['adm_id'] == region, '10percentile_obs_yield'] = np.percentile(stats_region['Yield'],10)
        df_forecast.loc[df_forecast['adm_id'] == region, '90percentile_obs_yield'] = np.percentile(stats_region['Yield'], 90)
        df_forecast.loc[df_forecast['adm_id'] == region, 'fyield_diff_pct (last 5 yrs in data avail)'] = \
            100 * ((fyield_region - stats_region['Yield'][-5::].mean()) / stats_region['Yield'][-5::].mean()  )  # ADDED LAST 5 YEARS
        df_forecast.loc[df_forecast['adm_id'] == region, 'avg_obs_area_last5yrs'] = stats_region['Area'][-5::].mean()
        df_forecast.loc[df_forecast['adm_id'] == region, 'fproduction_percentile'] = \
            percentile_below(stats_region['Production'], df_forecast.loc[df_forecast['adm_id'] == region, 'fproduction(fyield*avg_obs_area_last5yrs)'].values)

    forecast_fn = os.path.join(config.ope_run_out_dir, datetime.datetime.today().strftime('%Y%m%d') + '_' +
                               uset['crop'] + '_forecast_month_season_' + str(uset['forecast_time'])
                               + '_issue_early_' + str(forecast_issue_calendar_month) + '_' + uset['algorithm'] +
                               '.csv' )
    df_forecast.to_csv(forecast_fn, float_format='%.2f')


def combine_models(fns):
    # find best model by admin
    # compare best model and peak and adjust extreme predictions, if any
    # Avoid very low or high estimates
    # For very low
    # if y_prct of best is <= 0.1 and peak have a larger one, take that of peak
    # For high estimates
    # .. same on 0.9 percentile
    # check if fyield > max_obs_yield or fyield < min_obs_yield, in such cases use min or max

    listDFs = []
    for fn in fns:
        listDFs.append(pd.read_csv(fn, index_col=0))
    df = pd.concat(listDFs, axis=0, ignore_index=True)
    # get best by admin
    df_best = df.loc[df.groupby('adm_id')['fyield_rRMSEp_prct_hindcasting'].idxmin()]

    # adjust high low est that are below or above 0.1 and 0.9 perct
    df_conservative_estimates = df.iloc[:0]
    list_replace = []
    for adm_id in df['adm_id'].unique():
        df_adm = df[df['adm_id'] == adm_id]
        df_best_adm = df_best[df_best['adm_id'] == adm_id]
        if df_best_adm['algorithm'].values[0] == 'PeakNDVI':
            # no option, keep it
            df_conservative_estimates = pd.concat([df_conservative_estimates, df_best_adm])
            list_replace.append('best')
        else:
            # check fyield_percentile < 0.1
            if df_best_adm['fyield_percentile'].values[0] < 0.1:
                # if peak exists (in case of few obs we may have ML but no peak) has a larger one use it
                if len(df_adm[df_adm['algorithm'] == 'PeakNDVI']) > 1 and df_adm[df_adm['algorithm'] == 'PeakNDVI']['fyield_percentile'].values[0] >= 0.1:
                    df_conservative_estimates = pd.concat([df_conservative_estimates, df_adm[df_adm['algorithm'] == 'PeakNDVI']])
                    list_replace.append('replaced because yield_percentile<0.1')
                else:
                    df_conservative_estimates = pd.concat([df_conservative_estimates, df_best_adm])
                    list_replace.append('best, but yield_percentile<0.1 that could not be replaced with Peak')
            # check fyield_percentile > 0.9
            elif df_best_adm['fyield_percentile'].values[0] > 0.9:
                # if peak has a smaller one use it
                if len(df_adm[df_adm['algorithm'] == 'PeakNDVI']) > 1 and df_adm[df_adm['algorithm'] == 'PeakNDVI']['fyield_percentile'].values[0] <= 0.9:
                    df_conservative_estimates = pd.concat([df_conservative_estimates, df_adm[df_adm['algorithm'] == 'PeakNDVI']])
                    list_replace.append('replaced because yield_percentile>0.9')
                else:
                    df_conservative_estimates = pd.concat([df_conservative_estimates, df_best_adm])
                    list_replace.append('best, but yield_percentile>0.9 that could not be replaced with Peak')
            else:
                df_conservative_estimates = pd.concat([df_conservative_estimates, df_best_adm])
                list_replace.append('best')
    df_conservative_estimates['Consolidation_log'] = list_replace

    # I have the df_conservative_estimates, now check that estimates do not exceed min max, for percetmile
    use = 'percentiles' # 'minmax' or 'percentiles'

    # min
    if use == 'minmax':
        column_low = 'min_obs_yield'
        column_high = 'max_obs_yield'
    elif use == 'percentiles':
        column_low = '10percentile_obs_yield'
        column_high = '90percentile_obs_yield'
    else:
        exit('use not defined')
    mask = df_conservative_estimates['fyield'] < df_conservative_estimates[column_low]
    if len(df_conservative_estimates[mask]) > 0:
        # replace estimate with max
        df_conservative_estimates.loc[mask, 'fyield'] = df_conservative_estimates[mask][column_low]
        if use == 'minmax':
            df_conservative_estimates.loc[mask, 'fyield_percentile'] = 0
            df_conservative_estimates.loc[mask, 'Consolidation_log'] = 'fyield < min, reset to min'
        elif use == 'percentiles':
            df_conservative_estimates.loc[mask, 'fyield_percentile'] = df_conservative_estimates[mask][column_low]
            df_conservative_estimates.loc[mask, 'Consolidation_log'] = 'fyield < 10percentile, reset to 10percentile'
        else:
            exit('use not defined')

        df_conservative_estimates.loc[mask, 'fyield_diff_pct (last 5 yrs in data avail)'] = \
            100 * (df_conservative_estimates[mask]['fyield'] - df_conservative_estimates[mask]['avg_obs_yield_last5yrs']) / \
            df_conservative_estimates[mask]['avg_obs_yield_last5yrs']
        df_conservative_estimates.loc[mask, 'fproduction(fyield*avg_obs_area_last5yrs)'] = df_conservative_estimates[mask]['fyield'] * \
                                                                             df_conservative_estimates[mask][
                                                                                 'avg_obs_area_last5yrs']
        df_conservative_estimates.loc[mask, 'fproduction_percentile'] = np.nan

    # max
    mask = df_conservative_estimates['fyield'] > df_conservative_estimates[column_high]
    if len(df_conservative_estimates[mask]) > 0:
        # replace estimate with max
        df_conservative_estimates.loc[mask, 'fyield'] = df_conservative_estimates[mask][column_high]
        if use == 'minmax':
            df_conservative_estimates.loc[mask, 'fyield_percentile'] = 1
            df_conservative_estimates.loc[mask, 'Consolidation_log'] = 'fyield > max, reset to max'
        elif use == 'percentiles':
            df_conservative_estimates.loc[mask, 'fyield_percentile'] = df_conservative_estimates[mask][column_high]
            df_conservative_estimates.loc[mask, 'Consolidation_log'] = 'fyield > 90percentile, reset to 90percentile'
        else:
            exit('use not defined')
        df_conservative_estimates.loc[mask, 'fyield_percentile'] = 1
        df_conservative_estimates.loc[mask, 'fyield_diff_pct (last 5 yrs in data avail)'] = \
            100 * (df_conservative_estimates[mask]['fyield'] - df_conservative_estimates[mask]['avg_obs_yield_last5yrs']) / df_conservative_estimates[mask]['avg_obs_yield_last5yrs']
        df_conservative_estimates.loc[mask, 'fproduction(fyield*avg_obs_area_last5yrs)'] = df_conservative_estimates[mask]['fyield'] * df_conservative_estimates[mask]['avg_obs_area_last5yrs']
        df_conservative_estimates.loc[mask, 'fproduction_percentile'] =  np.nan


    return df_best, df_conservative_estimates

def make_consolidated_ope(config):
    """
    Gather crop-specific and unit level forecasts and generate a nation-scale forecast
    """
    # get pipeline specific forecast files, all crops here
    fns = [x for x in glob.glob(os.path.join(config.ope_run_out_dir, '*.csv')) if 'national' not in x and 'consolidated' not in x and 'best_by' not in x]
    # get yield stats
    df_stats = b101_load_cleaned.LoadCleanedLabel(config)

    crop_list, yields, yieldsdiff, production, ppercentile = [], [], [], [], []
    dict_list = {'crop_list': [], 'yields': [], 'yieldsdiff': [], 'production': [], 'ppercentile': []}
    dict4nat = {'best_accuracy': copy.deepcopy(dict_list), 'conservative_estimates': copy.deepcopy(dict_list)}

    # for crop_name in crop_Names: # by crop
    for crop_name in config.crops:
        print(crop_name)
        fns_crop = [s for s in fns if re.search(f".*{crop_name}_forecast.*", s)]
        # first save a single file with all estimations, ordered by region and by fyield_rMAEp_hindcasting
        listDFs = []
        # for fn in fns:
        for fn in fns_crop:
            listDFs.append(pd.read_csv(fn, index_col=0))
        df = pd.concat(listDFs, axis=0, ignore_index=True)
        df = df.sort_values(['Crop_name', 'Region_name', 'algorithm', 'fyield_rRMSEp_prct_hindcasting'], ascending=[True, True, True, True])
        # replace Null_model with NullModel to avoid wrong splitting
        tmp = fns_crop[0].replace("Null_model", "NullModel")
        fn_out = '_'.join(tmp.split('_')[0:-1] + ['unconsolidated.csv'])
        fn_out = os.path.join(config.ope_run_out_dir, 'best_accuracy', os.path.basename(fn_out))
        Path(os.path.join(config.ope_run_out_dir, 'best_accuracy')).mkdir(parents=True, exist_ok=True)
        df.to_csv(fn_out, index=False)

        # best model by admin (best accuracy and conservative accuracy)
        df_best_est, df_cons_est = combine_models(fns_crop)
        fn_out = '_'.join(tmp.split('_')[0:-1] + ['best_by_admin.csv'])
        fn_out = os.path.join(config.ope_run_out_dir, 'best_accuracy', os.path.basename(fn_out))
        df_best_est = df_best_est.sort_values(by=['Region_name'], ascending=True)
        df_best_est.to_csv(fn_out, index=False)

        # update conservative estimates
        fn_out = '_'.join(tmp.split('_')[0:-1] + ['conservative_estimates.csv'])
        # Subset stats
        df_stats_i = df_stats[df_stats['Crop_name'] == crop_name]
        df_stats_i = df_stats_i[df_stats_i['adm_name'].isin(df_cons_est['Region_name'])]

        # Recompute production percentile (because if min or max are used, they do not have percentiles?)
        for region in df_cons_est['Region_name']:
            if df_cons_est.loc[df_cons_est['Region_name'] == region, 'fproduction_percentile'].isna().sum() == 1:
                stats_region = df_stats_i[df_stats_i['adm_name'] == region]
                df_cons_est.loc[df_cons_est['Region_name'] == region, 'fproduction_percentile'] = \
                    percentile_below(stats_region['Production'],
                    df_cons_est.loc[df_cons_est['Region_name'] == region, 'fproduction(fyield*avg_obs_area_last5yrs)'].values)
        # Save updated values
        fn_out = os.path.join(config.ope_run_out_dir, 'conservative_estimates', os.path.basename(fn_out))
        Path(os.path.join(config.ope_run_out_dir, 'conservative_estimates')).mkdir(parents=True, exist_ok=True)
        df_cons_est = df_cons_est.sort_values(by=['Region_name'], ascending=True)
        df_cons_est.to_csv(fn_out, index=False)
        #df_cons_est.to_csv(fn_out)
        for selection_type, df in zip(['best_accuracy', 'conservative_estimates'], [df_best_est, df_cons_est]):
            dirName = os.path.join(config.ope_run_out_dir, selection_type)
            # compute national stats
            prod = df['fproduction(fyield*avg_obs_area_last5yrs)'].sum()
            area = df['avg_obs_area_last5yrs'].sum()
            # Define a lambda function to compute the weighted mean:
            wm = lambda x: np.average(x, weights=df_stats_i.loc[x.index, "Area"])
            # Note: the weighted sum of Yi, is exactly the same of Y_nat = Prod_nat / Area_nat
            df_stats_sum = df_stats_i.groupby(['Year']).agg(
                Production=pd.NamedAgg(column="Production", aggfunc="sum"),
                nat_yield=pd.NamedAgg(column="Yield", aggfunc=wm)
            )

            crop_list.append(crop_name)
            dict4nat[selection_type]['crop_list'].append(crop_name)
            production.append(prod)
            dict4nat[selection_type]['production'].append(prod)
            yields.append(prod / area)
            dict4nat[selection_type]['yields'].append(prod / area)
            yieldsdiff.append(100 * ((prod / area) - df_stats_sum['nat_yield'][-5:].mean()) / df_stats_sum['nat_yield'][-5:].mean())
            dict4nat[selection_type]['yieldsdiff'].append(100 * ((prod / area) - df_stats_sum['nat_yield'][-5:].mean()) / df_stats_sum['nat_yield'][-5:].mean())
            ppercentile.append(percentile_below(df_stats_sum['Production'], prod))
            dict4nat[selection_type]['ppercentile'].append(percentile_below(df_stats_sum['Production'], prod))

            # map consolidated and text about national production
            national_text = crop_name + ', ' + config.country_name_in_shp_file + ', aggregated yield forecast (area weighted) = ' + \
                            str(np.round(prod / area, 2)) + ', \n % difference with last avail. 5 years = ' + \
                            str(np.round(100 * ((prod / area) - df_stats_sum['nat_yield'][-5:].mean()) / df_stats_sum[
                                'nat_yield'][-5:].mean(), 2))
            e110_ope_figs.map(df, config, '', dirName, config.fn_reference_shape,
                              config.country_name_in_shp_file, title=national_text, suffix='_' + selection_type)
    # get elements of filename
    tmp = fns_crop[0].replace("Null_model", "NullModel")
    fn_out = '_'.join(tmp.split('_')[0:-1])
    for selection_type in ['best_accuracy', 'conservative_estimates']:
        dirName = os.path.join(config.ope_run_out_dir, selection_type)
        df = pd.DataFrame({'Crop_Name': dict4nat[selection_type]['crop_list'],
                           'fyield': dict4nat[selection_type]['yields'],
                           'yield_diff_pct': dict4nat[selection_type]['yieldsdiff'],
                           'fproduction(fyield*avg_obs_area_last5yrs)': dict4nat[selection_type]['production'],
                           'fproduction_percentile':  dict4nat[selection_type]['ppercentile']})

        fn_parts = os.path.basename(fn_out).split('_')
        fn_parts[1] = 'national-forecast'
        fn_parts.append(selection_type)
        filename = os.path.join(dirName, '_'.join(fn_parts)+'.csv')
        df.to_csv(filename)




# def combine_models_consistency_check(fns):
#     # compare best model and peak and adjust extreme predictions, if any
#     # 2024 09 23, Mic removed LASSO as it is still ML and may not be selected in fast tuning
#     # Avoid very low or high estimates
#     # For very low
#     # if y_prct of best is <= 0.1 and peak have a larger one, take that of peak
#     # For high estimates
#     # .. same on 0.9 percentile
#     # check if still y_percent < 0.1  --> min  Yield
#     # or  > 0.9 and -- > max Yield
#     # in that case use min or max observed yield as forecasted yield
#
#     # first plot all model resulst
#     # load all results
#     listDFs = []
#     for fn in fns:
#         listDFs.append(pd.read_csv(fn, index_col=0))
#     df = pd.concat(listDFs, axis=0, ignore_index=True)
#
#     if len([x for x in fns if 'PeakNDVI' not in x]) > 0: # there is a best model that is not Peak
#         # get the name of the non peak
#         fn_1 = [x for x in fns if 'PeakNDVI' not in x][0]
#         df_best = pd.read_csv(fn_1, index_col=0)
#         #get peak
#         fn_3 = [x for x in fns if 'PeakNDVI' in x][0]
#         df_peak = pd.read_csv(fn_3, index_col=0)
#
#         # non peak may have been run on a subset of admin units. If some are missing, drop them on peak
#         df_peak = df_peak[df_peak['adm_id'].isin(df_best['adm_id'])]
#
#         # Avoid very low estimates
#
#         # if y_prct of best (can be best or lasso) is  <= 0.1 and peak have a larger one, take that of peak and put in best
#         df_best.loc[(df_best.fyield_percentile <= 0.1) & (df_best.fyield_percentile < df_peak.fyield_percentile), :] = \
#             df_peak.loc[(df_best.fyield_percentile <= 0.1) & (df_best.fyield_percentile < df_peak.fyield_percentile), :]
#         # Avoid very high estimates
#         # same on 0.9 percentile]
#         df_best.loc[(df_best.fyield_percentile >= 0.9) & (df_best.fyield_percentile > df_peak.fyield_percentile), :] = \
#             df_peak.loc[(df_best.fyield_percentile >= 0.9) & (df_best.fyield_percentile > df_peak.fyield_percentile), :]
#     elif len([x for x in fns if  'PeakNDVI' not in x]) == 0: # there is only  Peak
#         fn_2 = [x for x in fns if 'PeakNDVI' in x][0]
#         df_best = pd.read_csv(fn_2, index_col=0)
#     else:
#         print('Inconsistent number of files. End of the world. Stopping now')
#         AssertionError
#     # check if still y_percent < 0.1  --> min  Yield or  > 0.9 -- > max Yield
#     if sum(df_best.fyield_percentile <= 0.1) > 0:
#         # min
#         select_rows_min = (df_best.fyield < df_best.min_obs_yield) & (df_best.fyield_percentile <= 0.1)
#         df_best.loc[select_rows_min, 'fyield'] = df_best.loc[select_rows_min, 'min_obs_yield']
#         # update variables related to fyield
#         df_best.loc[select_rows_min, ['fyield_percentile']] = 0
#         df_best.loc[select_rows_min, ['algorithm']] = 'min'
#         df_best.loc[select_rows_min, ['fyield_SD_Bootstrap_1yr', 'cv_mae']] = np.nan
#
#         df_best.loc[select_rows_min, 'yield_diff_pct'] = 100 * (df_best.loc[select_rows_min, 'fyield'] -
#                                                              df_best.loc[select_rows_min, 'avg_obs_yield']) / \
#                                                       df_best.loc[select_rows_min, 'avg_obs_yield']
#         df_best.loc[select_rows_min, 'fproduction(fyield*avg_obs_area_last5yrs)'] = \
#             df_best.loc[select_rows_min, 'fyield'] * df_best.loc[select_rows_min, 'avg_obs_area_last5yrs']
#         df_best.loc[select_rows_min, 'fproduction_percentile'] = np.nan
#
#         # max
#         select_rows_max = (df_best.fyield > df_best.max_obs_yield) & (df_best.fyield_percentile >= 0.9)
#         df_best.loc[select_rows_max, 'fyield'] = df_best.loc[select_rows_max, 'max_obs_yield']
#         # update variables related to fyield
#         df_best.loc[select_rows_max, ['fyield_percentile']] = 0
#         df_best.loc[select_rows_max, ['algorithm']] = 'max'
#         df_best.loc[select_rows_max, ['fyield_SD_Bootstrap_1yr', 'cv_mae']] = np.nan
#
#         df_best.loc[select_rows_max, 'yield_diff_pct'] = 100 * (df_best.loc[select_rows_max, 'fyield'] -
#                                                              df_best.loc[select_rows_max, 'avg_obs_yield']) / \
#                                                       df_best.loc[select_rows_max, 'avg_obs_yield']
#         df_best.loc[select_rows_max, 'fproduction(fyield*avg_obs_area_last5yrs)'] = \
#             df_best.loc[select_rows_max, 'fyield'] * df_best.loc[select_rows_max, 'avg_obs_area_last5yrs']
#         df_best.loc[select_rows_max, 'fproduction_percentile'] = np.nan
#
#     return df_best
# def combine_models_consistency_check_old_with_LASSO(fns):
#     # compare best model, lasso and peak and adjust extreme predictions, if any
#     # Avoid very low or high estimates
#     # For very low
#     # if y_prct of best is <= 0.1 and lasso have a larger one, take that of lasso and put in best
#     # if y_prct of new best (can be best or lasso) is still <= 0.1 and peak have a larger one, take that of peak and put in best
#     # For high estimates
#     # .. same on 0.9 percentile
#     # check if still y_percent < 0.1  --> min  Yield
#     # or  > 0.9 and -- > max Yield
#     # in that case use min or max observed yield as forecasted yield
#
#     # first plot all model resulst
#     # load all results
#     listDFs = []
#     for fn in fns:
#         listDFs.append(pd.read_csv(fn, index_col=0))
#     df = pd.concat(listDFs, axis=0, ignore_index=True)
#
#     if len([x for x in fns if 'Lasso' not in x and 'PeakNDVI' not in x]) > 0: # there is a best model that is not Lasso or Peak
#         # get the name of the non lasso and non peak
#         fn_1 = [x for x in fns if 'Lasso' not in x and 'PeakNDVI' not in x][0]
#         df_best = pd.read_csv(fn_1, index_col=0)
#         #get lasso
#         fn_2 = [x for x in fns if 'Lasso' in x][0]
#         df_lasso = pd.read_csv(fn_2, index_col=0)
#         #get peak
#         fn_3 = [x for x in fns if 'PeakNDVI' in x][0]
#         df_peak = pd.read_csv(fn_3, index_col=0)
#
#         # Avoid very low estimates
#         # if y_prct of best is <= 0.1 and lasso have a larger one, take that of lasso and put in best
#         df_best.loc[(df_best.fyield_percentile <= 0.1) & (df_best.fyield_percentile < df_lasso.fyield_percentile), :] =\
#             df_lasso.loc[(df_best.fyield_percentile <= 0.1) & (df_best.fyield_percentile < df_lasso.fyield_percentile), :]
#         # if y_prct of new best (can be best or lasso) is still <= 0.1 and peak have a larger one, take that of lasso and put in best
#         df_best.loc[(df_best.fyield_percentile <= 0.1) & (df_best.fyield_percentile < df_peak.fyield_percentile), :] = \
#             df_peak.loc[(df_best.fyield_percentile <= 0.1) & (df_best.fyield_percentile < df_peak.fyield_percentile), :]
#         # Avoid very high estimates
#         # same on 0.9 percentile
#         df_best.loc[(df_best.fyield_percentile >= 0.9) & (df_best.fyield_percentile > df_lasso.fyield_percentile), :] = \
#             df_lasso.loc[(df_best.fyield_percentile >= 0.9) & (df_best.fyield_percentile > df_lasso.fyield_percentile), :]
#         df_best.loc[(df_best.fyield_percentile >= 0.9) & (df_best.fyield_percentile > df_peak.fyield_percentile), :] = \
#             df_peak.loc[(df_best.fyield_percentile >= 0.9) & (df_best.fyield_percentile > df_peak.fyield_percentile), :]
#     elif len([x for x in fns if 'Lasso' not in x and 'PeakNDVI' not in x]) == 0: # there is only Lasso or Peak
#         # same if lasso was already best
#         fn_1 = [x for x in fns if 'Lasso' in x][0]
#         df_best = pd.read_csv(fn_1, index_col=0)
#
#         fn_2 = [x for x in fns if 'PeakNDVI' in x][0]
#         df_lasso = pd.read_csv(fn_2, index_col=0)
#         df_best.loc[(df_best.fyield_percentile <= 0.1) & (df_best.fyield_percentile < df_lasso.fyield_percentile), :] = \
#             df_lasso.loc[(df_best.fyield_percentile <= 0.1) & (df_best.fyield_percentile < df_lasso.fyield_percentile), :]
#         df_best.loc[(df_best.fyield_percentile >= 0.9) & (df_best.fyield_percentile > df_lasso.fyield_percentile), :] = \
#             df_lasso.loc[(df_best.fyield_percentile >= 0.9) & (df_best.fyield_percentile > df_lasso.fyield_percentile), :]
#     else:
#         print('Inconsistent number of files. End of the world. Stopping now')
#         AssertionError
#     # check if still y_percent < 0.1  --> min  Yield or  > 0.9 -- > max Yield
#     if sum(df_best.fyield_percentile <= 0.1) > 0:
#         # min
#         select_rows_min = (df_best.fyield < df_best.min_obs_yield) & (df_best.fyield_percentile <= 0.1)
#         df_best.loc[select_rows_min, 'fyield'] = df_best.loc[select_rows_min, 'min_obs_yield']
#         # update variables related to fyield
#         df_best.loc[select_rows_min, ['fyield_percentile']] = 0
#         df_best.loc[select_rows_min, ['algorithm']] = 'min'
#         df_best.loc[select_rows_min, ['fyield_SD_Bootstrap_1yr', 'cv_mae']] = np.nan
#
#         df_best.loc[select_rows_min, 'yield_diff_pct'] = 100 * (df_best.loc[select_rows_min, 'fyield'] -
#                                                              df_best.loc[select_rows_min, 'avg_obs_yield']) / \
#                                                       df_best.loc[select_rows_min, 'avg_obs_yield']
#         df_best.loc[select_rows_min, 'fproduction(fyield*avg_obs_area_last5yrs)'] = \
#             df_best.loc[select_rows_min, 'fyield'] * df_best.loc[select_rows_min, 'avg_obs_area_last5yrs']
#         df_best.loc[select_rows_min, 'fproduction_percentile'] = np.nan
#
#         # max
#         select_rows_max = (df_best.fyield > df_best.max_obs_yield) & (df_best.fyield_percentile >= 0.9)
#         df_best.loc[select_rows_max, 'fyield'] = df_best.loc[select_rows_max, 'max_obs_yield']
#         # update variables related to fyield
#         df_best.loc[select_rows_max, ['fyield_percentile']] = 0
#         df_best.loc[select_rows_max, ['algorithm']] = 'max'
#         df_best.loc[select_rows_max, ['fyield_SD_Bootstrap_1yr', 'cv_mae']] = np.nan
#
#         df_best.loc[select_rows_max, 'yield_diff_pct'] = 100 * (df_best.loc[select_rows_max, 'fyield'] -
#                                                              df_best.loc[select_rows_max, 'avg_obs_yield']) / \
#                                                       df_best.loc[select_rows_max, 'avg_obs_yield']
#         df_best.loc[select_rows_max, 'fproduction(fyield*avg_obs_area_last5yrs)'] = \
#             df_best.loc[select_rows_max, 'fyield'] * df_best.loc[select_rows_max, 'avg_obs_area_last5yrs']
#         df_best.loc[select_rows_max, 'fproduction_percentile'] = np.nan
#
#     return df_best


# def to_csv_old(config, forecast_issue_calendar_month, uset, regions, forecasts, rMAE_p_hindcasting, runID=''):
#     df_forecast = pd.DataFrame({'adm_id': np.nan,
#                                 'adm_id': regions,
#                                 'Region_name': np.nan,
#                                 'Crop_name': uset['crop'],
#                                 'fyield': forecasts,
#                                 # 'fyield_SD_Bootstrap_1yr': funcertainty,
#                                 'fyield_rMAEp_hindcasting': rMAE_p_hindcasting,
#                                 'fyield_percentile': np.nan,
#                                 'avg_obs_yield': np.nan,
#                                 'avg_obs_yield_last5yrs': np.nan,
#                                 'min_obs_yield': np.nan,
#                                 'max_obs_yield': np.nan,
#                                 'fyield_diff_pct (last 5 yrs in data avail)': np.nan,
#                                 'avg_obs_area_last5yrs': np.nan,
#                                 'fproduction(fyield*avg_obs_area_last5yrs)': np.nan,
#                                 'fproduction_percentile': np.nan,
#                                 'algorithm': uset['algorithm'],
#                                 'runID': runID})
#
#     # get yield stats
#     stats = b101_load_cleaned.LoadCleanedLabel(config)
#     stats = stats[stats['Crop_name'] == uset['crop']]
#
#     for region in regions:
#         # get stats for region and sort by year (*)to take last 5)
#         stats_region = stats[stats['adm_id'] == region].sort_values(by=['Year'])
#         fyield_region = df_forecast.loc[df_forecast['adm_id'] == region, 'fyield'].values
#         df_forecast.loc[df_forecast['adm_id'] == region, 'adm_id'] = stats_region.iloc[0]['adm_id']
#         df_forecast.loc[df_forecast['adm_id'] == region, 'fyield_percentile'] = \
#             percentile_below(stats_region['Yield'], fyield_region)
#         df_forecast.loc[df_forecast['adm_id'] == region, 'Region_name'] = \
#             stats_region.iloc[0]['adm_name']
#         df_forecast.loc[df_forecast['adm_id'] == region, 'fproduction(fyield*avg_obs_area_last5yrs)'] = \
#             df_forecast.loc[df_forecast['adm_id'] == region, 'fyield'] * stats_region['Area'][-5::].mean()
#         df_forecast.loc[df_forecast['adm_id'] == region, 'avg_obs_yield'] = stats_region['Yield'].mean()
#         df_forecast.loc[df_forecast['adm_id'] == region, 'avg_obs_yield_last5yrs'] = stats_region['Yield'][-5::].mean() #avg_obs_yield_last5yrs
#         df_forecast.loc[df_forecast['adm_id'] == region, 'min_obs_yield'] = stats_region['Yield'].min()
#         df_forecast.loc[df_forecast['adm_id'] == region, 'max_obs_yield'] = stats_region['Yield'].max()
#         df_forecast.loc[df_forecast['adm_id'] == region, 'fyield_diff_pct (last 5 yrs in data avail)'] = \
#             100 * ((fyield_region - stats_region['Yield'][-5::].mean()) / stats_region['Yield'][-5::].mean()  )  # ADDED LAST 5 YEARS
#         df_forecast.loc[df_forecast['adm_id'] == region, 'avg_obs_area_last5yrs'] = stats_region['Area'][-5::].mean()
#         df_forecast.loc[df_forecast['adm_id'] == region, 'fproduction_percentile'] = \
#             percentile_below(stats_region['Production'], df_forecast.loc[df_forecast['adm_id'] == region, 'fproduction(fyield*avg_obs_area_last5yrs)'].values)
#
#     forecast_fn = os.path.join(config.ope_run_out_dir, datetime.datetime.today().strftime('%Y%m%d') + '_' +
#                                uset['crop'] + '_forecast_month_season_' + str(uset['forecast_time'])
#                                + '_issue_early_' + str(forecast_issue_calendar_month) + '_' + uset['algorithm'] +
#                                '.csv' )
#     df_forecast.to_csv(forecast_fn, float_format='%.2f')
