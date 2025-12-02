import pandas as pd
import numpy as np
import os
import copy
import re
from B_preprocess import b101_load_cleaned
from E_viz import e110_ope_figs
from F_post_processsing import F100_analyze_hindcast_output
from B_preprocess import b50_yield_data_analysis
import datetime
import glob
from pathlib import Path
import json





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


def to_csv(config, forecast_issue_calendar_month, uset, regions, forecasts, runID=''):
    df_forecast = pd.DataFrame({'adm_id': regions,
                                'Region_name': np.nan,
                                'Crop_name': uset['crop'],
                                'fyield': forecasts,
                                'fyield_RMSEp_hindcasting': np.nan,
                                'fyield_rRMSEp_prct_hindcasting': np.nan,
                                'fyield_MAEp_hindcasting': np.nan,
                                'fyield_rMAEp_prct_hindcasting': np.nan,
                                'fyield_r2_coeff_det_by_admin_hindcasting': np.nan,
                                'fyield_r2_pearson_by_admin_hindcasting': np.nan,
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
        # benchmark models are not panel, so if an admin have few data there might be no model (in hindcasting, in ope is computed anyhow)
        # 10 is the bare min in hindcasting (hardcoded in d100 line 63 (5 if mulipolygons)
        if len(defAuError_region) == 0:
            #try setting everythin to nan except region id and name
            df_forecast.loc[df_forecast['adm_id'] == region, 'Region_name'] = stats_region.iloc[0]['adm_name']
            df_forecast.loc[df_forecast['adm_id'] == region, [col for col in df_forecast.columns if col not in ['adm_id', 'df_forecast']]] = np.nan
        else:
            fyield_region = df_forecast.loc[df_forecast['adm_id'] == region, 'fyield'].values
            # get hindcasting stats also including those for NASA Intercomparison
            df_forecast.loc[df_forecast['adm_id'] == region, 'fyield_RMSEp_hindcasting'] =  defAuError_region['rmse'].values[0]
            df_forecast.loc[df_forecast['adm_id'] == region, 'fyield_rRMSEp_prct_hindcasting'] = defAuError_region['rrmse_prct'].values[0]
            df_forecast.loc[df_forecast['adm_id'] == region, 'fyield_MAEp_hindcasting'] =   defAuError_region['mae'].values[0]
            df_forecast.loc[df_forecast['adm_id'] == region, 'fyield_rMAEp_prct_hindcasting'] =  defAuError_region['rmae_prct'].values[0]
            df_forecast.loc[df_forecast['adm_id'] == region, 'fyield_r2_coeff_det_by_admin_hindcasting'] = defAuError_region['r2_coeff_det'].values[0]
            df_forecast.loc[df_forecast['adm_id'] == region, 'fyield_r2_pearson_by_admin_hindcasting'] =   defAuError_region['r2_pearson'].values[0]

            df_forecast.loc[df_forecast['adm_id'] == region, 'adm_id'] = stats_region.iloc[0]['adm_id']
            df_forecast.loc[df_forecast['adm_id'] == region, 'fyield_percentile'] = \
                percentile_below(stats_region['Yield'], fyield_region)
            df_forecast.loc[df_forecast['adm_id'] == region, 'Region_name'] = stats_region.iloc[0]['adm_name']
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
                               uset['crop'] + '_forecast_mInSeas' + str(uset['forecast_time'])
                               + '_early_' + str(forecast_issue_calendar_month) + '_' + uset['algorithm'] +
                               '.csv' )
    df_forecast.to_csv(forecast_fn, float_format='%.3f')

def check_replace_percentile_exceedance(df_in, column_low, column_high, column_yield):
    """
    This function check that values in column_yield of df do not exceed percentile value in
    column_low, column_high. It can b used in ope but also in hindcasting error at natianl scale (national_error_hindcasting in F100)
    """
    df = df_in.copy()
    # in case it is coming from hindacasting it has the log already (computed at ope stage), remove
    if 'Consolidation_log' in df.columns:
        df['Consolidation_log'] = ''

    # min
    mask = df[column_yield] < df[column_low]
    if len(df[mask]) > 0:
        # replace estimate with min
        df.loc[mask, column_yield] = df[mask][column_low]
        df.loc[mask, 'fyield_percentile'] = df[mask][column_low]
        df.loc[mask, 'Consolidation_log'] = 'fyield < 10percentile, reset to 10percentile'
        df.loc[mask, 'fyield_diff_pct (last 5 yrs in data avail)'] = \
            100 * (df[mask][column_yield] - df[mask][
                'avg_obs_yield_last5yrs']) / df[mask]['avg_obs_yield_last5yrs']
        df.loc[mask, 'fproduction(fyield*avg_obs_area_last5yrs)'] = \
            df[mask][column_yield] * df[mask]['avg_obs_area_last5yrs']
        df.loc[mask, 'fproduction_percentile'] = np.nan

    # max
    mask = df[column_yield] > df[column_high]
    if len(df[mask]) > 0:
        # replace estimate with max
        df.loc[mask, column_yield] = df[mask][column_high]
        df.loc[mask, 'fyield_percentile'] = df[mask][column_high]
        df.loc[mask, 'Consolidation_log'] = 'fyield > 90percentile, reset to 90percentile'
        # df.loc[mask, 'fyield_percentile'] = 1
        df.loc[mask, 'fyield_diff_pct (last 5 yrs in data avail)'] = \
            100 * (df[mask][column_yield] - df[mask][
                'avg_obs_yield_last5yrs']) / df[mask]['avg_obs_yield_last5yrs']
        df.loc[mask, 'fproduction(fyield*avg_obs_area_last5yrs)'] = \
            df[mask][column_yield] * df[mask]['avg_obs_area_last5yrs']
        df.loc[mask, 'fproduction_percentile'] = np.nan
    return df
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
    # Michele 2025 05 21 remove selection of peak, keep only perecentiles
    # df_conservative_estimates = df.iloc[:0]
    df_conservative_estimates = df_best.copy()
    df_conservative_estimates['Consolidation_log'] = ''
    # list_replace = []
    # for adm_id in df['adm_id'].unique():
    #     df_adm = df[df['adm_id'] == adm_id]
    #     df_best_adm = df_best[df_best['adm_id'] == adm_id]
    #     if df_best_adm['algorithm'].values[0] == 'PeakNDVI':
    #         # no option, keep it
    #         df_conservative_estimates = pd.concat([df_conservative_estimates, df_best_adm])
    #         list_replace.append('best')
    #     else:
    #         # check fyield_percentile < 0.1
    #         if df_best_adm['fyield_percentile'].values[0] < 0.1:
    #             # if peak exists (in case of few obs we may have ML but no peak) has a larger one use it
    #             if len(df_adm[df_adm['algorithm'] == 'PeakNDVI']) > 1 and df_adm[df_adm['algorithm'] == 'PeakNDVI']['fyield_percentile'].values[0] >= 0.1:
    #                 df_conservative_estimates = pd.concat([df_conservative_estimates, df_adm[df_adm['algorithm'] == 'PeakNDVI']])
    #                 list_replace.append('replaced because yield_percentile<0.1')
    #             else:
    #                 df_conservative_estimates = pd.concat([df_conservative_estimates, df_best_adm])
    #                 list_replace.append('best, but yield_percentile<0.1 that could not be replaced with Peak')
    #
    #         # check fyield_percentile > 0.9
    #         elif df_best_adm['fyield_percentile'].values[0] > 0.9:
    #             # if peak has a smaller one use it
    #             if len(df_adm[df_adm['algorithm'] == 'PeakNDVI']) > 1 and df_adm[df_adm['algorithm'] == 'PeakNDVI']['fyield_percentile'].values[0] <= 0.9:
    #                 df_conservative_estimates = pd.concat([df_conservative_estimates, df_adm[df_adm['algorithm'] == 'PeakNDVI']])
    #                 list_replace.append('replaced because yield_percentile>0.9')
    #             else:
    #                 df_conservative_estimates = pd.concat([df_conservative_estimates, df_best_adm])
    #                 list_replace.append('best, but yield_percentile>0.9 that could not be replaced with Peak')
    #         else:
    #             df_conservative_estimates = pd.concat([df_conservative_estimates, df_best_adm])
    #             list_replace.append('best')
    # df_conservative_estimates['Consolidation_log'] = list_replace

    # I have the df_conservative_estimates, now check that estimates do not exceed min max, or percentile
    # ALWAYS USE PERCENTILES
    column_low = '10percentile_obs_yield'
    column_high = '90percentile_obs_yield'
    # use = 'percentiles' # 'minmax' or 'percentiles'
    # # min
    # if use == 'minmax':
    #     column_low = 'min_obs_yield'
    #     column_high = 'max_obs_yield'
    # elif use == 'percentiles':
    #     column_low = '10percentile_obs_yield'
    #     column_high = '90percentile_obs_yield'
    # else:
    #     exit('use not defined')
    df_conservative_estimates = check_replace_percentile_exceedance(df_conservative_estimates, column_low, column_high, 'fyield')



    return df_best, df_conservative_estimates

def NASA_format(df_in, config):
    pd.set_option('display.max_columns', None)
    df = df_in.copy()
    dir_NASA = os.path.join(config.ope_run_out_dir, "best_accuracy", "NasaHarvest_format")
    Path(dir_NASA).mkdir(parents=True, exist_ok=True)
    # First make forecast according to template
    df.rename(columns={'adm_id': 'source_id'}, inplace=True)  # rename adm_id
    fn = b50_yield_data_analysis.find_last_version_csv(config.AOI + '_STATS', config.data_dir)
    df.insert(loc=1, column='source_name_version', value=fn)
    df.insert(loc=2, column='admin_0', value=config.country_name_in_shp_file)
    df.rename(columns={'Region_name': 'admin_1'}, inplace=True)
    df.insert(loc=4, column='admin_2', value="")
    df.insert(loc=5, column='admin_3', value="")
    df.insert(loc=6, column='planted_year', value=config.harvest_year+config.plantingYearDelta)
    df.insert(loc=7, column='approx_planted_month', value=config.sosMonth)
    df.insert(loc=8, column='harvest_year', value=config.harvest_year)
    df.insert(loc=9, column='approx_harvest_month', value=config.eosMonth)
    df.rename(columns={'Crop_name': 'crop'}, inplace=True)
    df.insert(loc=11, column='crop_season', value="n.a.")
    df.insert(loc=12, column='forecast_issue_date (yyyy-mm-dd)', value=pd.Timestamp.now().strftime("%Y-%m-%d"))
    analysisOutputDir = os.path.join(config.models_out_dir, 'Analysis')
    with open(os.path.join(analysisOutputDir, 'date_model_tune.json'), 'r') as fp:
        date_run = json.load(fp)
    df.insert(loc=13, column='date_model_run (yyyy-mm-dd)', value=date_run['date_run'])
    df.insert(loc=14, column='input_croptype_product', value='n.a.')
    df.insert(loc=15, column='group', value='JRC')
    df.insert(loc=16, column='model_version', value='mil4cast_' + date_run['date_run'])
    # yield_fcst (tn_ha) from fyield
    df.rename(columns={'fyield': 'yield_fcst (tn_ha)'}, inplace=True)
    df.insert(loc=18, column='is_final', value='n.a.')
    df.insert(loc=19, column='notes', value='')
    df_forecast = df.iloc[:, :20].copy()
    fn_out = os.path.join(dir_NASA, 'JRC_' + config.AOI + '_forecast_' + datetime.date.today().strftime("%Y-%m-%d") + '.csv')
    df_forecast.to_csv(fn_out, index=False)

        # remove columns not requested
    df = df.drop(columns=["yield_fcst (tn_ha)", "is_final", "notes"])
    df.rename(columns={'crop_season': 'season'}, inplace=True)
    df.insert(loc=17, column='cross_validation', value=0)
    df.rename(columns={'fyield_MAEp_hindcasting': 'yield_mae'}, inplace=True)
    df.rename(columns={'fyield_RMSEp_hindcasting': 'yield_rmse'}, inplace=True)
    df.rename(columns={'fyield_rRMSEp_prct_hindcasting': 'yield_rrmse'}, inplace=True)
    df.rename(columns={'fyield_rMAEp_prct_hindcasting': 'yield_mape'}, inplace=True)
    df.rename(columns={'fyield_r2_pearson_by_admin_hindcasting': 'yield_r2_pearson'}, inplace=True)
    df.rename(columns={'fyield_r2_coeff_det_by_admin_hindcasting': 'yield_r2_true'}, inplace=True)
    # reorder according to template
    df = df[['source_id', 'source_name_version', 'admin_0', 'admin_1', 'admin_2', 'admin_3', 'planted_year', 'approx_planted_month', 'harvest_year', 'approx_harvest_month', 'crop', 'season',
            'forecast_issue_date (yyyy-mm-dd)', 'date_model_run (yyyy-mm-dd)', 'input_croptype_product', 'group', 'model_version', 'cross_validation',
            'yield_mae', 'yield_rmse', 'yield_rrmse', 'yield_mape', 'yield_r2_pearson', 'yield_r2_true']]
    df["metric_years"] = ",".join(str(y) for y in range(config.year_start, config.year_end+1))
    df["notes"] = ""
    fn_out = os.path.join(dir_NASA, 'JRC_' + config.AOI + '_accuracy_' + datetime.date.today().strftime("%Y-%m-%d") + '.csv')
    df.to_csv(fn_out, index=False)



def make_consolidated_ope(config):
    """
    Gather crop-specific and unit level forecasts and generate a nation-scale forecast
    """
    # get pipeline specific forecast files, all crops here
    fns = [x for x in glob.glob(os.path.join(config.ope_run_out_dir, '*.csv')) if 'national' not in x and 'consolidated' not in x and 'best_by' not in x]
    # get yield stats
    df_stats = b101_load_cleaned.LoadCleanedLabel(config)

    # crop_list, yields, yieldsdiff, production, ppercentile, hind_rmse,  hind_rrmse_prct, hind_r2 = [], [], [], [], [], [], [], []
    # dict_list = {'crop_list': [], 'yields': [], 'yieldsdiff': [], 'production': [], 'ppercentile': []}
    dict_list = {'crop_list': [], 'yields': [], 'yieldsdiff': [], 'production': [], 'ppercentile': [],
                 'hind_rmse': [], 'hind_rrmse_prct': [], 'hind_r2': []}
    dict4nat = {'best_accuracy': copy.deepcopy(dict_list), 'conservative_estimates': copy.deepcopy(dict_list)}

    # for crop_name in crop_Names: # by crop
    for crop_name in config.crops:
        print(crop_name)
        fns_crop = [s for s in fns if re.search(f".*{crop_name}_forecast.*", s)]
        # first save a single file with all estimations (unconsolidated), ordered by region and by fyield_rMAEp_hindcasting
        listDFs = []
        # for fn in fns:
        for fn in fns_crop:
            listDFs.append(pd.read_csv(fn, index_col=0))
        df = pd.concat(listDFs, axis=0, ignore_index=True)
        df = df.sort_values(['Crop_name', 'Region_name', 'algorithm', 'fyield_rRMSEp_prct_hindcasting'], ascending=[True, True, True, True])
        # replace Null_model with NullModel to avoid wrong splitting
        tmp = fns_crop[0].replace("Null_model", "NullModel")
        fn_out = '_'.join(tmp.split('_')[0:-1] + ['unconsolidated.csv'])
        Path(os.path.join(config.ope_run_out_dir, 'best_accuracy')).mkdir(parents=True, exist_ok=True)
        fn_out = os.path.join(config.ope_run_out_dir, 'best_accuracy', os.path.basename(fn_out))
        # Write unconsolidated
        df.to_csv(fn_out, index=False)

        # best model by admin (best accuracy and conservative accuracy)
        df_best_est, df_cons_est = combine_models(fns_crop)
        fn_out = '_'.join(tmp.split('_')[0:-1] + ['best_by_admin.csv'])
        fn_out = os.path.join(config.ope_run_out_dir, 'best_accuracy', os.path.basename(fn_out))
        df_best_est = df_best_est.sort_values(by=['Region_name'], ascending=True)
        # Write consolidated (best by admin) and add NASA harvest Intercomparison format
        df_best_est.to_csv(fn_out, index=False)
        # NASA PART
        NASA_format(df_best_est, config)
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
        # Write conservative
        df_cons_est.to_csv(fn_out, index=False)
        # get National level estimates
        for selection_type, df in zip(['best_accuracy', 'conservative_estimates'], [df_best_est, df_cons_est]):
            # crop_list.append(crop_name)
            dict4nat[selection_type]['crop_list'].append(crop_name)
            # Forecasted yield, production and area
            prod = df['fproduction(fyield*avg_obs_area_last5yrs)'].sum()
            area = df['avg_obs_area_last5yrs'].sum()
            # production.append(prod)
            dict4nat[selection_type]['production'].append(prod)
            # This national level yield obtained by prod/area, ii equal to area weighted sum of yields, y = prod/area = S(y*a) / S(a)
            # yields.append(prod / area)
            dict4nat[selection_type]['yields'].append(prod / area)
            # now get some stats from official stats to position current forecasts in historical perspective
            # Define a lambda function to compute the weighted mean (using area as weighting factor) of variable:
            wm = lambda x: np.average(x, weights=df_stats_i.loc[x.index, "Area"])
            # Note: the weighted sum of Yi, is exactly the same of Y_nat = Prod_nat / Area_nat
            df_stats_sum = df_stats_i.groupby(['Year']).agg(
                Production=pd.NamedAgg(column="Production", aggfunc="sum"),
                nat_yield=pd.NamedAgg(column="Yield", aggfunc=wm))

            # yieldsdiff.append(100 * ((prod / area) - df_stats_sum['nat_yield'][-5:].mean()) / df_stats_sum['nat_yield'][-5:].mean())
            dict4nat[selection_type]['yieldsdiff'].append(100 * ((prod / area) - df_stats_sum['nat_yield'][-5:].mean()) / df_stats_sum['nat_yield'][-5:].mean())
            # ppercentile.append(percentile_below(df_stats_sum['Production'], prod))
            dict4nat[selection_type]['ppercentile'].append(percentile_below(df_stats_sum['Production'], prod))

            # map consolidated and text about national production
            national_text = crop_name + ', ' + config.country_name_in_shp_file + ', aggregated yield forecast (area weighted) = ' + \
                            str(np.round(prod / area, 2)) + ', \n % difference with last avail. 5 years = ' + \
                            str(np.round(100 * ((prod / area) - df_stats_sum['nat_yield'][-5:].mean()) / df_stats_sum[
                                'nat_yield'][-5:].mean(), 2))
            dirName = os.path.join(config.ope_run_out_dir, selection_type)
            e110_ope_figs.map(df, config, '', dirName, config.fn_reference_shape,
                              config.country_name_in_shp_file, title=national_text, suffix='_' + selection_type)
            # here df tel me which model (and model run) is used for each admin. using mRes I can get national level errors
            rmseNat, rrmseNatprct, r2Nat = F100_analyze_hindcast_output.national_error_hindcasting(df, dirName,config, selection_type, df_stats_sum)
            # #'hind_rmse': [], 'hind_rrmse_prct': [], 'hind_r2': []
            dict4nat[selection_type]['hind_rmse'].append(rmseNat)
            dict4nat[selection_type]['hind_rrmse_prct'].append(rrmseNatprct)
            dict4nat[selection_type]['hind_r2'].append(r2Nat)
            # get elements of filename
    tmp = fns_crop[0].replace("Null_model", "NullModel")
    fn_out = '_'.join(tmp.split('_')[0:-1])
    for selection_type in ['best_accuracy', 'conservative_estimates']:
        dirName = os.path.join(config.ope_run_out_dir, selection_type)
        df = pd.DataFrame({'Crop_Name': dict4nat[selection_type]['crop_list'],
                           'fyield': dict4nat[selection_type]['yields'],
                           'yield_diff_pct': dict4nat[selection_type]['yieldsdiff'],
                           'fproduction(fyield*avg_obs_area_last5yrs)': dict4nat[selection_type]['production'],
                           'fproduction_percentile':  dict4nat[selection_type]['ppercentile'],
                           'RMSE_hindcasting': dict4nat[selection_type]['hind_rmse'],
                           'rRMSE_prct_hindcasting': dict4nat[selection_type]['hind_rrmse_prct'],
                           'R2_hindcasting': dict4nat[selection_type]['hind_r2']})

        fn_parts = os.path.basename(fn_out).split('_')
        fn_parts[1] = 'national'#'national-forecast'
        fn_parts.append(selection_type)
        filename = os.path.join(dirName, '_'.join(fn_parts)+'.csv')
        df.to_csv(filename)

