import pathlib
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import textwrap
from A_config import a10_config
from E_viz import e100_eval_figs
from D_modelling import d090_model_wrapper, d140_modelStats
from D_modelling import d140_modelStats
from F_post_processsing import F110_process_opeForecast_output
from B_preprocess import b101_load_cleaned
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
    # get best n ML configurations by lead time, crop type and y var PLUS Tabl
    # Tab change 2025
    ben2discard = list(filter(lambda x: x != "Tab", mlsettings.benchmarks))
    #moML = mo[mo['Estimator'].isin(mlsettings.benchmarks) == False]
    moML = mo[mo['Estimator'].isin(ben2discard) == False]
    bn = moML.groupby(['Crop', var4time]).apply(
        lambda x: x.sort_values([metric2use], ascending=sortAscending).head(n)).reset_index(drop=True)
    # always add the benchmarks

    bn = bn.sort_values([var4time, 'Crop', metric2use], \
                        ascending=[True, True, sortAscending])
    # bn.to_csv(analysisOutputDir + '/' + 'ML_models_to_run_with_tuning.csv', index=False)
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


    # Work on run outputs (avg performances)
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
    moML = mo[mo['Estimator'].isin(mlsettings.benchmarks) == False]
    b1ML = moML.groupby(['Crop', var4time]).apply(
        lambda x: x.sort_values([metric2use], ascending=sortAscending).head(1)).reset_index(drop=True)
    # always add the benchmarks
    tmp = mo.groupby(['Crop', var4time]) \
        .apply(lambda x: x.loc[x['Estimator'].isin(mlsettings.benchmarks)]).reset_index(drop=True)
    tmp = tmp.drop_duplicates(subset=[var4time, 'Estimator', 'Crop'])
    b1 = pd.concat([b1ML, tmp])
    b1.to_csv(analysisOutputDir + '/' + 'all_model_best1.csv', index=False)

    # Work at tha AU level
    # compute error at AU level

    # Marocco has boundary changing over time, it is a special case, I want to keep errors only on the last set of boundaries
    if config.AOI == 'MA':
        print('**************************************************************')
        print('Special case with chenging bounadriies')
        print('cosi non va prende ids multipli ')
        print('**************************************************************')
        # open the cleaned stats file to determine which are the admin ids connected to the last shape
        s = b101_load_cleaned.LoadCleanedLabel(config)
        # Get the first record with the maximum 'Year'
        first_max_year_record = s.loc[s['Year'] == s['Year'].max()].head(1)
        last_shp = first_max_year_record['Ref_shp'].iloc[0]
        # now get all the admin ids that are using this shp
        s = s[s['Ref_shp'] == last_shp]
        adm_id_in_shp_2keep = s['adm_id'].to_list()
        #add a suffix to results
        suffix = '_last_shp'
    else:
        adm_id_in_shp_2keep = None
        suffix = ''

    b1withAUerror = e100_eval_figs.AU_error(b1, config, analysisOutputDir, suffix, adm_id_in_shp_2keep=adm_id_in_shp_2keep)
    # in order to assign the same colors and keep a defined order I have to do have a unique label for all ML models
    b1withAUerror['tmp_est'] = b1withAUerror['Estimator'].map(lambda x: x if x in mlsettings.benchmarks else 'ML')




    # plot scatter of Ml and bechmark, one plot per crop and forecasting time (this function is saving mRes)

    e100_eval_figs.scatter_plots_and_maps(b1withAUerror, config, mlsettings, var4time, analysisOutputDir, fn_shape_gaul1,
                                          country_name_in_shp_file, gdf_gaul0_column=gdf_gaul0_column)
    # add best by admin
    # take best (whatever, by admin)
    b1AU = b1withAUerror.copy().reset_index(drop=True)
    bestByAU = b1AU.loc[b1AU.groupby(['adm_id', 'forecast_time', 'Crop_ID|'])['rmse'].idxmin()]
    # of this hybrid best, compute avg rrmse_prct and avg rrmse_prct weighted by area
    # to be comparable with that of the single models, I have to compute it the same way,
    # through d140_modelStats.allStats_spatial(mRes) and using as mRES the mixture of different models
    adm_id_for_best = -900 # I need to assign a fake id (later I drop rows with duplicate id)
    for crop in bestByAU['Crop'].unique():
        for ft in bestByAU['forecast_time'].unique():
            hybrid_mRes = pd.DataFrame()
            cropFtDf = bestByAU.loc[(bestByAU['Crop'] == crop) & (bestByAU['forecast_time'] == ft)]
            # now for each estimator present, I have to get the corresponding mRES and cherry pick only where it is best
            for est in cropFtDf['Estimator'].unique():
                # get admin ids where this est is best
                admWhreIsBest = cropFtDf[cropFtDf['Estimator'] == est]['adm_id'].to_list()
                # get run_id
                runID = cropFtDf.loc[cropFtDf['Estimator'] == est]['runID'].iloc[0]
                myID = f'{runID:06d}'
                fn_mRes_out = os.path.join(config.models_out_dir, 'ID_' + str(myID) +
                                           '_crop_' + crop + '_Yield_' + est + '_mres.csv')
                mRes = pd.read_csv(fn_mRes_out)
                mResToRetain = mRes.loc[mRes['adm_id'].isin(admWhreIsBest)]
                hybrid_mRes = pd.concat([hybrid_mRes, mResToRetain])
            res = d140_modelStats.allStats_overall(hybrid_mRes)
            mResWithArea = hybrid_mRes.merge(cropFtDf[['adm_id|', 'Area|mean']], how='left', left_on='adm_id',
                                             right_on='adm_id|')
            resw = d140_modelStats.rmse_rrmse_weighed_overall(hybrid_mRes, mResWithArea['Area|mean'])
            # add it to b1
            tmp = pd.DataFrame([{'Estimator': 'BestByAdmin', 'tmp_est': 'BestByAdmin', 'rRMSE_p': res['rel_Pred_RMSE'],
                                 'rRMSE_p_areaWeighted': resw['rel_Pred_RMSE'],
                                 'forecast_time': ft, 'Crop': crop,
                                 'forecast_issue_calendar_month': cropFtDf['forecast_issue_calendar_month'].iloc[0],
                                 'runID': adm_id_for_best}])
            adm_id_for_best = adm_id_for_best -1
            b1withAUerror = pd.concat([b1withAUerror, tmp], ignore_index=True)
    # save main statistical indicators indicators in a csv file
    e100_eval_figs.summary_stats(b1withAUerror, config, var4time, analysisOutputDir)
    # plot it by forecasting time (simple bars for each forecasting time), mRes is created above
    e100_eval_figs.bars_by_forecast_time2(b1withAUerror, config, 'rRMSE_p', mlsettings, var4time, analysisOutputDir)
    # e100_eval_figs.bars_by_forecast_time(b1, config, metric2use, mlsettings, var4time, analysisOutputDir)

    print('Compare output ended')


def national_error_hindcasting(df, dirName, config, selection_type, df_stats_sum):
    # here I am considering that for ope forecast we use a mix of models based which one if performing best in each subanat
    # admin. Using this info, I make for each year the national forecast and compute errors against available stats

    # get unique list of models (runIDs) used, I will need to fetch their mRes to get hindcasting by admin
    runIDlist = df.runID.unique()
    runIDlist = [int(x) for x in runIDlist]
    #for each year I have to compute national yield forecasted and get true national yield, both are in mres

    for m in runIDlist:
        try:
            myID = f'{m:06d}'
        except:
            print()
        crop = df['Crop_name'].iloc[0]
        est = df.loc[df['runID'] == m, 'algorithm'].iloc[0]
        fn_mRes_out = os.path.join(config.models_out_dir, 'ID_' + str(myID) +
                                   '_crop_' + crop + '_Yield_' + est +
                                   '_mres.csv')
        mRes = pd.read_csv(fn_mRes_out)
        mRes['runID'] = m
        if m == runIDlist[0]:
            # first entry, get columns and make empty df to store all the needed mRes
            mResAll = pd.DataFrame(columns=list(mRes.columns))

        mResAll = pd.concat([mResAll, mRes], ignore_index=True)

    years = mResAll['Year'].unique()
    mResNat = pd.DataFrame(columns=['Year', 'Y_est', 'Y_obs'])
    for yr in years:
        mResAllYr = mResAll[mResAll['Year'] == yr]
        dfYr = pd.merge(df, mResAllYr, on=['runID', 'adm_id'], how='left')
        # prepare a df to store all dfYr
        if yr == years[0]:
            # first entry, make empty df to store all the years
            dfAllYr = pd.DataFrame(columns=list(dfYr.columns))
        # be care here for conservative to intercept percentile truncation!!
        if selection_type == 'conservative_estimates':
            column_low = '10percentile_obs_yield'
            column_high = '90percentile_obs_yield'
            dfYr = F110_process_opeForecast_output.check_replace_percentile_exceedance(dfYr, column_low, column_high, 'yLoo_pred')
        dfAllYr = pd.concat([dfAllYr, dfYr], ignore_index=True)
        mask = np.logical_or(np.isnan(dfYr['yLoo_true']), np.isnan(dfYr['yLoo_pred'])) # mask both for a fair comparison
        ma = np.ma.MaskedArray(dfYr['yLoo_true'], mask=mask)
        Y_nat_obs = np.ma.average(ma, weights=dfYr['avg_obs_area_last5yrs'])
        # if all are masked, it returns masked, return nan instead
        if np.ma.is_masked(Y_nat_obs):
            Y_nat_obs = np.nan
        ma = np.ma.MaskedArray(dfYr['yLoo_pred'], mask=mask)
        Y_nat_est = np.ma.average(ma, weights=dfYr['avg_obs_area_last5yrs'])
        if np.ma.is_masked(Y_nat_est):
            Y_nat_est = np.nan
        mResNat.loc[len(mResNat)] = [yr, Y_nat_est, Y_nat_obs]
    # Compute rmse and rrmse
    rmseNat = d140_modelStats.rmse_nan(mResNat['Y_est'], mResNat['Y_obs'])
    rrmseNatprct = rmseNat / mResNat['Y_obs'].mean()*100
    # 2025 07 01, error in passing data to r2 score, first is true, second is estimated
    r2Nat = d140_modelStats.r2_nan(mResNat['Y_obs'], mResNat['Y_est'])
    # r2Nat = d140_modelStats.r2_nan(mResNat['Y_est'], mResNat['Y_obs'])
    mResNat = mResNat.sort_values(by='Year')
    # make a time plot
    plt.figure(figsize=(5, 3))
    plt.plot(mResNat['Year'], mResNat['Y_est'], "-o", color='green', label="Predicted")
    plt.plot(mResNat['Year'], mResNat['Y_obs'], "-o", color='black', label="Observed")
    plt.legend(prop={'size': 6})
    plt.xticks(np.arange(min(mResNat['Year']-1), max(mResNat['Year']+1), 1), rotation=40)
    units = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_measurement_units.csv'))

    plt.ylabel("Yield [" + units['Yield'].iloc[0] + "]")
    title = 'Crop: ' + crop + ". Ope type: " + selection_type + ". Hindcasting: R2 = " + str(round(r2Nat, 2)) + ", RMSE = " + str(round(rmseNat, 2)) + ", rRMSE % = " + str(round(rrmseNatprct, 2))
    title = '\n'.join(textwrap.wrap(title, 40))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(dirName, crop + '_' + selection_type + '_national_hincasting.png'))
    plt.close()

    #save data plot and dfYr for all years
    mResNat.to_csv(os.path.join(dirName, crop + '_' + selection_type + '_national_hincasting.csv'), index=False)
    dfAllYr = dfAllYr.drop(columns=['fyield', 'fyield_percentile', 'fyield_diff_pct (last 5 yrs in data avail)', 'fproduction(fyield*avg_obs_area_last5yrs)', 'fproduction_percentile'])
    dfAllYr.to_csv(os.path.join(dirName, crop + '_' + selection_type + '_all_hindacsting_results.csv'), index=False)

    return rmseNat, rrmseNatprct, r2Nat


