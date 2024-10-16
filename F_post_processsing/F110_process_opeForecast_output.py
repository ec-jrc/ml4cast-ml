import pandas as pd
import numpy as np
import os
from B_preprocess import b100_load
from E_viz import e110_ope_figs
import datetime
import glob





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
def to_csv(config, forecast_issue_calendar_month, uset, regions, forecasts, rMAE_p_hindcasting, runID=''):
    df_forecast = pd.DataFrame({'adm_id': np.nan,
                                'adm_id': regions,
                                'Region_name': np.nan,
                                'Crop_name': uset['crop'],
                                'fyield': forecasts,
                                # 'fyield_SD_Bootstrap_1yr': funcertainty,
                                'fyield_rMAEp_hindcasting': rMAE_p_hindcasting,
                                'fyield_percentile': np.nan,
                                'avg_obs_yield': np.nan,
                                'avg_obs_yield_last5yrs': np.nan,
                                'min_obs_yield': np.nan,
                                'max_obs_yield': np.nan,
                                'fyield_diff_pct (last 5 yrs in data avail)': np.nan,
                                'avg_obs_area_last5yrs': np.nan,
                                'fproduction(fyield*avg_obs_area_last5yrs)': np.nan,
                                'fproduction_percentile': np.nan,
                                'algorithm': uset['algorithm'],
                                'runID': runID})

    # get yield stats
    #stats = b100_load.LoadLabel_Exclude_Missing(config, save_csv = False, plot_fig= False, verbose= False)
    stats = b100_load.LoadLabel(config, save_csv=False, plot_fig=False)
    stats = stats[stats['Crop_name'] == uset['crop']]

    for region in regions:
        # get stats for region and sort by year (*)to take last 5)
        stats_region = stats[stats['adm_id'] == region].sort_values(by=['Year'])
        fyield_region = df_forecast.loc[df_forecast['adm_id'] == region, 'fyield'].values
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

def combine_models_consistency_check(fns):
    # compare best model and peak and adjust extreme predictions, if any
    # 2024 09 23, Mic removed LASSO as it is still ML and may not be selected in fast tuning
    # Avoid very low or high estimates
    # For very low
    # if y_prct of best is <= 0.1 and peak have a larger one, take that of peak
    # For high estimates
    # .. same on 0.9 percentile
    # check if still y_percent < 0.1  --> min  Yield
    # or  > 0.9 and -- > max Yield
    # in that case use min or max observed yield as forecasted yield

    # first plot all model resulst
    # load all results
    listDFs = []
    for fn in fns:
        listDFs.append(pd.read_csv(fn, index_col=0))
    df = pd.concat(listDFs, axis=0, ignore_index=True)

    if len([x for x in fns if 'PeakNDVI' not in x]) > 0: # there is a best model that is not Peak
        # get the name of the non peak
        fn_1 = [x for x in fns if 'PeakNDVI' not in x][0]
        df_best = pd.read_csv(fn_1, index_col=0)
        #get peak
        fn_3 = [x for x in fns if 'PeakNDVI' in x][0]
        df_peak = pd.read_csv(fn_3, index_col=0)

        # Avoid very low estimates

        # if y_prct of best (can be best or lasso) is  <= 0.1 and peak have a larger one, take that of peak and put in best
        df_best.loc[(df_best.fyield_percentile <= 0.1) & (df_best.fyield_percentile < df_peak.fyield_percentile), :] = \
            df_peak.loc[(df_best.fyield_percentile <= 0.1) & (df_best.fyield_percentile < df_peak.fyield_percentile), :]
        # Avoid very high estimates
        # same on 0.9 percentile]
        df_best.loc[(df_best.fyield_percentile >= 0.9) & (df_best.fyield_percentile > df_peak.fyield_percentile), :] = \
            df_peak.loc[(df_best.fyield_percentile >= 0.9) & (df_best.fyield_percentile > df_peak.fyield_percentile), :]
    elif len([x for x in fns if  'PeakNDVI' not in x]) == 0: # there is only  Peak
        fn_2 = [x for x in fns if 'PeakNDVI' in x][0]
        df_best = pd.read_csv(fn_2, index_col=0)
    else:
        print('Inconsistent number of files. End of the world. Stopping now')
        AssertionError
    # check if still y_percent < 0.1  --> min  Yield or  > 0.9 -- > max Yield
    if sum(df_best.fyield_percentile <= 0.1) > 0:
        # min
        select_rows_min = (df_best.fyield < df_best.min_obs_yield) & (df_best.fyield_percentile <= 0.1)
        df_best.loc[select_rows_min, 'fyield'] = df_best.loc[select_rows_min, 'min_obs_yield']
        # update variables related to fyield
        df_best.loc[select_rows_min, ['fyield_percentile']] = 0
        df_best.loc[select_rows_min, ['algorithm']] = 'min'
        df_best.loc[select_rows_min, ['fyield_SD_Bootstrap_1yr', 'cv_mae']] = np.nan

        df_best.loc[select_rows_min, 'yield_diff_pct'] = 100 * (df_best.loc[select_rows_min, 'fyield'] -
                                                             df_best.loc[select_rows_min, 'avg_obs_yield']) / \
                                                      df_best.loc[select_rows_min, 'avg_obs_yield']
        df_best.loc[select_rows_min, 'fproduction(fyield*avg_obs_area_last5yrs)'] = \
            df_best.loc[select_rows_min, 'fyield'] * df_best.loc[select_rows_min, 'avg_obs_area_last5yrs']
        df_best.loc[select_rows_min, 'fproduction_percentile'] = np.nan

        # max
        select_rows_max = (df_best.fyield > df_best.max_obs_yield) & (df_best.fyield_percentile >= 0.9)
        df_best.loc[select_rows_max, 'fyield'] = df_best.loc[select_rows_max, 'max_obs_yield']
        # update variables related to fyield
        df_best.loc[select_rows_max, ['fyield_percentile']] = 0
        df_best.loc[select_rows_max, ['algorithm']] = 'max'
        df_best.loc[select_rows_max, ['fyield_SD_Bootstrap_1yr', 'cv_mae']] = np.nan

        df_best.loc[select_rows_max, 'yield_diff_pct'] = 100 * (df_best.loc[select_rows_max, 'fyield'] -
                                                             df_best.loc[select_rows_max, 'avg_obs_yield']) / \
                                                      df_best.loc[select_rows_max, 'avg_obs_yield']
        df_best.loc[select_rows_max, 'fproduction(fyield*avg_obs_area_last5yrs)'] = \
            df_best.loc[select_rows_max, 'fyield'] * df_best.loc[select_rows_max, 'avg_obs_area_last5yrs']
        df_best.loc[select_rows_max, 'fproduction_percentile'] = np.nan

    return df_best
def combine_models_consistency_check_old_with_LASSO(fns):
    # compare best model, lasso and peak and adjust extreme predictions, if any
    # Avoid very low or high estimates
    # For very low
    # if y_prct of best is <= 0.1 and lasso have a larger one, take that of lasso and put in best
    # if y_prct of new best (can be best or lasso) is still <= 0.1 and peak have a larger one, take that of peak and put in best
    # For high estimates
    # .. same on 0.9 percentile
    # check if still y_percent < 0.1  --> min  Yield
    # or  > 0.9 and -- > max Yield
    # in that case use min or max observed yield as forecasted yield

    # first plot all model resulst
    # load all results
    listDFs = []
    for fn in fns:
        listDFs.append(pd.read_csv(fn, index_col=0))
    df = pd.concat(listDFs, axis=0, ignore_index=True)

    if len([x for x in fns if 'Lasso' not in x and 'PeakNDVI' not in x]) > 0: # there is a best model that is not Lasso or Peak
        # get the name of the non lasso and non peak
        fn_1 = [x for x in fns if 'Lasso' not in x and 'PeakNDVI' not in x][0]
        df_best = pd.read_csv(fn_1, index_col=0)
        #get lasso
        fn_2 = [x for x in fns if 'Lasso' in x][0]
        df_lasso = pd.read_csv(fn_2, index_col=0)
        #get peak
        fn_3 = [x for x in fns if 'PeakNDVI' in x][0]
        df_peak = pd.read_csv(fn_3, index_col=0)

        # Avoid very low estimates
        # if y_prct of best is <= 0.1 and lasso have a larger one, take that of lasso and put in best
        df_best.loc[(df_best.fyield_percentile <= 0.1) & (df_best.fyield_percentile < df_lasso.fyield_percentile), :] =\
            df_lasso.loc[(df_best.fyield_percentile <= 0.1) & (df_best.fyield_percentile < df_lasso.fyield_percentile), :]
        # if y_prct of new best (can be best or lasso) is still <= 0.1 and peak have a larger one, take that of lasso and put in best
        df_best.loc[(df_best.fyield_percentile <= 0.1) & (df_best.fyield_percentile < df_peak.fyield_percentile), :] = \
            df_peak.loc[(df_best.fyield_percentile <= 0.1) & (df_best.fyield_percentile < df_peak.fyield_percentile), :]
        # Avoid very high estimates
        # same on 0.9 percentile
        df_best.loc[(df_best.fyield_percentile >= 0.9) & (df_best.fyield_percentile > df_lasso.fyield_percentile), :] = \
            df_lasso.loc[(df_best.fyield_percentile >= 0.9) & (df_best.fyield_percentile > df_lasso.fyield_percentile), :]
        df_best.loc[(df_best.fyield_percentile >= 0.9) & (df_best.fyield_percentile > df_peak.fyield_percentile), :] = \
            df_peak.loc[(df_best.fyield_percentile >= 0.9) & (df_best.fyield_percentile > df_peak.fyield_percentile), :]
    elif len([x for x in fns if 'Lasso' not in x and 'PeakNDVI' not in x]) == 0: # there is only Lasso or Peak
        # same if lasso was already best
        fn_1 = [x for x in fns if 'Lasso' in x][0]
        df_best = pd.read_csv(fn_1, index_col=0)

        fn_2 = [x for x in fns if 'PeakNDVI' in x][0]
        df_lasso = pd.read_csv(fn_2, index_col=0)
        df_best.loc[(df_best.fyield_percentile <= 0.1) & (df_best.fyield_percentile < df_lasso.fyield_percentile), :] = \
            df_lasso.loc[(df_best.fyield_percentile <= 0.1) & (df_best.fyield_percentile < df_lasso.fyield_percentile), :]
        df_best.loc[(df_best.fyield_percentile >= 0.9) & (df_best.fyield_percentile > df_lasso.fyield_percentile), :] = \
            df_lasso.loc[(df_best.fyield_percentile >= 0.9) & (df_best.fyield_percentile > df_lasso.fyield_percentile), :]
    else:
        print('Inconsistent number of files. End of the world. Stopping now')
        AssertionError
    # check if still y_percent < 0.1  --> min  Yield or  > 0.9 -- > max Yield
    if sum(df_best.fyield_percentile <= 0.1) > 0:
        # min
        select_rows_min = (df_best.fyield < df_best.min_obs_yield) & (df_best.fyield_percentile <= 0.1)
        df_best.loc[select_rows_min, 'fyield'] = df_best.loc[select_rows_min, 'min_obs_yield']
        # update variables related to fyield
        df_best.loc[select_rows_min, ['fyield_percentile']] = 0
        df_best.loc[select_rows_min, ['algorithm']] = 'min'
        df_best.loc[select_rows_min, ['fyield_SD_Bootstrap_1yr', 'cv_mae']] = np.nan

        df_best.loc[select_rows_min, 'yield_diff_pct'] = 100 * (df_best.loc[select_rows_min, 'fyield'] -
                                                             df_best.loc[select_rows_min, 'avg_obs_yield']) / \
                                                      df_best.loc[select_rows_min, 'avg_obs_yield']
        df_best.loc[select_rows_min, 'fproduction(fyield*avg_obs_area_last5yrs)'] = \
            df_best.loc[select_rows_min, 'fyield'] * df_best.loc[select_rows_min, 'avg_obs_area_last5yrs']
        df_best.loc[select_rows_min, 'fproduction_percentile'] = np.nan

        # max
        select_rows_max = (df_best.fyield > df_best.max_obs_yield) & (df_best.fyield_percentile >= 0.9)
        df_best.loc[select_rows_max, 'fyield'] = df_best.loc[select_rows_max, 'max_obs_yield']
        # update variables related to fyield
        df_best.loc[select_rows_max, ['fyield_percentile']] = 0
        df_best.loc[select_rows_max, ['algorithm']] = 'max'
        df_best.loc[select_rows_max, ['fyield_SD_Bootstrap_1yr', 'cv_mae']] = np.nan

        df_best.loc[select_rows_max, 'yield_diff_pct'] = 100 * (df_best.loc[select_rows_max, 'fyield'] -
                                                             df_best.loc[select_rows_max, 'avg_obs_yield']) / \
                                                      df_best.loc[select_rows_max, 'avg_obs_yield']
        df_best.loc[select_rows_max, 'fproduction(fyield*avg_obs_area_last5yrs)'] = \
            df_best.loc[select_rows_max, 'fyield'] * df_best.loc[select_rows_max, 'avg_obs_area_last5yrs']
        df_best.loc[select_rows_max, 'fproduction_percentile'] = np.nan

    return df_best


def make_consolidated_ope(config):
    """
    Gather crop-specific and unit level forecasts and generate a nation-scale forecast
    """
    # get pipeline specific forecast files, all crops here
    fns = [x for x in glob.glob(os.path.join(config.ope_run_out_dir, '*.csv')) if 'national' not in x and 'consolidated' not in x]
    # get yield stats
    #df_stats = b100_load.LoadLabel_Exclude_Missing(config, save_csv=False, plot_fig=False, verbose=False)
    df_stats = b100_load.LoadLabel(config, save_csv=False, plot_fig=False)
    crop_list, production, ppercentile, yields, yieldsdiff = [], [], [], [], []
    # crop_Names = list(df_stats['Crop_name'].unique())
    # for crop_name in crop_Names: # by crop
    for crop_name in config.crops:
        print(crop_name)
        fns_crop = [x for x in fns if crop_name in x]
        # first save a signle file with all estimations, ordered by region and by fyield_rMAEp_hindcasting
        listDFs = []
        for fn in fns:
            listDFs.append(pd.read_csv(fn, index_col=0))
        df = pd.concat(listDFs, axis=0, ignore_index=True)
        df = df.sort_values(['Crop_name', 'Region_name', 'algorithm', 'fyield_rMAEp_hindcasting'], ascending=[True, True, True, True])
        fn_out = '_'.join(fns_crop[0].split('_')[0:-1] + ['unconsolidated.csv'])
        df.to_csv(fn_out, index=False)
        # here I have the models make the bar plot. x= different regions, for each reagion YF and YF diff with previous 5 years
        df_f = combine_models_consistency_check(fns_crop)
        fn_out = '_'.join(fns_crop[0].split('_')[0:-1] + ['consolidated.csv'])
        # Subset stats
        df_stats_i = df_stats[df_stats['Crop_name'] == crop_name]
        df_stats_i = df_stats_i[df_stats_i['adm_name'].isin(df_f['Region_name'])]

        # Recompute production percentile (because if min or max are used, they do not have percentiles?)
        for region in df_f['Region_name']:
            if df_f.loc[df_f['Region_name'] == region, 'fproduction_percentile'].isna().sum() == 1:
                stats_region = df_stats_i[df_stats_i['adm_name'] == region]
                df_f.loc[df_f['Region_name'] == region, 'fproduction_percentile'] = \
                    percentile_below(stats_region['Production'],
                    df_f.loc[df_f['Region_name'] == region, 'fproduction(fyield*avg_obs_area_last5yrs)'].values)
        # Save updated values
        df_f.to_csv(fn_out)
        # compute national stats
        prod = df_f['fproduction(fyield*avg_obs_area_last5yrs)'].sum()
        area = df_f['avg_obs_area_last5yrs'].sum()

        # Define a lambda function to compute the weighted mean:
        wm = lambda x: np.average(x, weights=df_stats_i.loc[x.index, "Area"])

        # Note: the weighted sum of Yi, is exactly the same of Y_nat = Prod_nat / Area_nat

        df_stats_sum = df_stats_i.groupby(['Year']).agg(
            Production=pd.NamedAgg(column="Production", aggfunc="sum"),
            nat_yield=pd.NamedAgg(column="Yield", aggfunc=wm)
        )

        crop_list.append(crop_name)
        production.append(prod)
        yields.append(prod / area)
        yieldsdiff.append(100 * ((prod / area) - df_stats_sum['nat_yield'].mean()) / df_stats_sum['nat_yield'].mean())
        ppercentile.append(percentile_below(df_stats_sum['Production'], prod))

        # map consolidated and text about national production
        national_text = crop_name + ',' + config.country_name_in_shp_file + ', aggregated yield forecast = ' + \
                        str(np.round(prod / area, 2)) + ', \n % difference with last avail. 5 years = '  + \
                        str(np.round(100 * ((prod / area) - df_stats_sum['nat_yield'].mean()) / df_stats_sum['nat_yield'].mean(),2))
        e110_ope_figs.map(df_f, config, '', config.ope_run_out_dir, config.fn_reference_shape,
                          config.country_name_in_shp_file, title=national_text)

    df = pd.DataFrame({'Crop_Name': crop_list,
                       'fyield': yields,
                       'yield_diff_pct': yieldsdiff,
                       'fproduction(fyield*avg_obs_area_last5yrs)': production,
                       'fproduction_percentile': ppercentile})


    fn_parts = os.path.basename(fn_out).split('_')
    fn_parts[1] = 'national-forecast'
    filename = os.path.join(config.ope_run_out_dir, '_'.join(fn_parts))
    df.to_csv(filename)
    # Make the map by admin of: consolidated yield forecast, diff % with 5 yrs avg, and print the national yiel/prod, and 5 yrs diff
