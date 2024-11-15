import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from B_preprocess import b100_load
from E_viz import e50_yield_data_analysis


def saveYieldStats(config, prct2retain=100):
    '''Compute statistics and save csv of the country stats
       - keep only from year of interest for the admin level stats'''
    # prct2retain is the percentage to retain
    if prct2retain > 100:
        print('requested percentage gt 100')
        exit()
    outDir = os.path.join(config.data_dir, 'Label_analysis' + str(prct2retain))
    Path(outDir).mkdir(parents=True, exist_ok=True)
    pd.set_option('display.max_columns', None)
    desired_width=10000
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns',100)

    # qaulity check and outlier removal
    stat_file = os.path.join(config.data_dir, config.AOI + '_STATS.csv')
    stats = b100_load.LoadLabel(stat_file, config.year_start, config.year_end, make_charts=True, perc_threshold=-1, crops_names=config.crops)

    units = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_measurement_units.csv'))
    area_unit = units['Area'].values[0]
    yield_unit = units['Yield'].values[0]

    # keep only from year of interest for the admin level stats
    stats = stats[stats['Year'] >= config.year_start]

    # Missing data are now dropped, this below is always empty
    # #save a file with missing data (all regions, no matter 90% production or not)
    # tmp = stats.copy()
    # tmp['Null'] = tmp['Yield'].isnull()
    # tmp = tmp[tmp['Null'] == True]
    # tmp = tmp.sort_values(by=['Crop_name','adm_id','Year'])
    # if len(tmp.index)>0:
    #     print('Missing records are present, inspect ' + os.path.join(config.data_dir, config.AOI + '_missing_data.csv'))
    # tmp.to_csv(os.path.join(outDir, config.AOI + '_missing_data.csv'), index=False)

    # Last5yrs stats
    if False:
        region_ids = stats['adm_id'].unique()
        crop_ids = stats['Crop_ID'].unique()
        statsLast5yrs = stats.copy()
        for rid in region_ids:
            for cid in crop_ids:
                lastYear = statsLast5yrs[(statsLast5yrs.adm_id == rid) & (statsLast5yrs.Crop_ID == cid)]['Year'].max()
                statsLast5yrs = statsLast5yrs.drop(statsLast5yrs[(statsLast5yrs.adm_id == rid) & \
                                                                 (statsLast5yrs.Crop_ID == cid) & \
                                                                 (statsLast5yrs.Year <= lastYear-5)].index)

        # Mean by: Region, Crop
        x5 = statsLast5yrs.groupby(['adm_id', 'Crop_ID']). \
            agg({'adm_name': 'first', 'Crop_name': 'first', 'Production': ['mean', 'std'],
                 'Yield': ['mean', 'std', 'count'], 'Area': ['mean', 'std']})
        # sort by production
        x5 = x5.sort_values(by=['Crop_ID', ('Production', 'mean')], ascending=False)
        # add an index 0, 1, ..
        x5.reset_index(inplace=True)

        # Compute, by crop  in all regions
        x5[('Crop_sum_production', '')] = x5.groupby('Crop_ID')[[('Production', 'mean')]].transform('sum')
        x5[('Cum_sum_production', '')] = x5.groupby('Crop_ID')[[('Production', 'mean')]].transform('cumsum')
        x5[('Perc_production', '')] = x5[('Production', 'mean')] / x5[('Crop_sum_production', '')] * 100
        x5[('Cum_perc_production', '')] = x5[('Cum_sum_production', '')] / x5[('Crop_sum_production', '')] * 100

        # and by region by crop
        x5[('Crop_sum_area', '')] = x5.groupby('adm_id')[[('Area', 'mean')]].transform('sum')
        x5[('Perc_area', '')] = x5[('Area', 'mean')] / x5[('Crop_sum_area', '')] * 100

        # keep a copy with all for later
        x05 = x5.copy(deep=True)
        # keep only the largest up to prct2retain production, by crop
        crops = x5['Crop_name', 'first'].unique()
        for c in crops:
            tmp = x5[x5['Crop_name', 'first'] == c]
            tmp = tmp.reset_index(drop=True)
            tmp = tmp.sort_values(by='Cum_perc_production')
            if prct2retain != 100:
                ind = tmp[tmp['Cum_perc_production'] >= prct2retain].index[0]
                tmp = tmp.iloc[0:ind + 1]

            x5 = x5.drop(x5[x5['Crop_name', 'first'] == c].index)
            x5 = pd.concat([x5, tmp])
        # remove multi column
        y5 = x5.copy()
        y5.columns = y5.columns.map(lambda v: '|'.join([str(i) for i in v]))
        y5.to_csv(os.path.join(outDir, config.AOI + '_5yrsStats_retainPRCT' + str(prct2retain) + '.csv'), index=False)

    # LT stats
    #Mean by: Region, Crop
    x = stats.groupby(['adm_id', 'Crop_ID']). \
         agg({'adm_name':'first','Crop_name':'first','Production': ['mean', 'std'], 'Yield': ['mean', 'std', 'count'], 'Area': ['mean', 'std']})

    # sort by area
    x = x.sort_values(by=['Crop_ID', ('Area', 'mean')], ascending=False)
    # add an index 0, 1, ..
    x.reset_index(inplace=True)

    # Compute, by crop in all regions
    x[('Crop_tot_area', '')] = x.groupby('Crop_ID')[[('Area', 'mean')]].transform('sum')
    x[('Crop_perc_area', '')] = x[('Area', 'mean')] / x[('Crop_tot_area', '')] * 100
    x[('Crop_cum_area', '')] = x.groupby('Crop_ID')[[('Area', 'mean')]].transform('cumsum')
    x[('Crop_perc_cum_area', '')] = x[('Crop_cum_area','')] / x[('Crop_tot_area', '')] * 100

    x[('Crop_total_production', '')] = x.groupby('Crop_ID')[[('Production', 'mean')]].transform('sum')
    x[('Crop_perc_production', '')] = x[('Production', 'mean')] / x[('Crop_total_production', '')] * 100

    # keep only the largest up to prct2retain area, by crop
    crops = x['Crop_name', 'first'].unique()
    for c in crops:
        tmp = x[x['Crop_name','first'] == c]
        tmp = tmp.reset_index(drop=True)
        tmp = tmp.sort_values(by='Crop_perc_cum_area')
        if prct2retain != 100:
            ind = tmp[tmp['Crop_perc_cum_area'] >= prct2retain].index[0]
            tmp = tmp.iloc[0:ind + 1]

        x = x.drop(x[x['Crop_name', 'first'] == c].index)
        x = pd.concat([x, tmp])
    # remove multi column
    y = x.copy()
    y.columns = y.columns.map(lambda v: '|'.join([str(i) for i in v]))
    y.to_csv(os.path.join(outDir, config.AOI + '_LTstats_retainPRCT' + str(prct2retain) + '.csv'), index=False)

    # save a cleaned stat file with the prct2retain to be retained

    # empty df but same columns, same dtypes, and no row
    stats_prct2retain = stats.iloc[:0,:].copy()
    for c in crops:
        adm_id_2retain_for_crop = y[y['Crop_name|first'] == c]['adm_id|'].unique()
        tmp = stats[(stats['Crop_name'] == c) & (stats['adm_id'].isin(adm_id_2retain_for_crop))]
        stats_prct2retain= pd.concat([stats_prct2retain, tmp])
    cleaned_prct2retain_file = stat_file.replace('.csv', '_cleaned' + str(prct2retain) + '.csv')
    stats_prct2retain.to_csv(cleaned_prct2retain_file, index=False)

    # bar plot of production, area and yield by region of retained statistics
    crops = x['Crop_name','first'].unique()
    for c in crops:
        # production
        xc = x[x['Crop_name','first'] == c]
        xc = xc.sort_values(by=[('Area', 'mean')], ascending=False)
        xdata = list(range(len(xc['adm_name'])))
        if area_unit == 'ha' and yield_unit == 't/ha':
            divider = 1000
        elif area_unit == 'ha' and yield_unit == 'kg/ha':
            divider = 1000000
        else:
            print('Measurement units not foreseen')
            exit()

        xc['Production', 'mean'] = xc['Production','mean']/divider
        xc['Production', 'std'] = xc['Production','std']/divider
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs = axs.flatten()
        xc.columns = xc.columns.map(lambda v: '|'.join([str(i) for i in v]))
        # plot production
        e50_yield_data_analysis.barDfColumn(xdata, xc, 'Production|mean', xc['adm_name|first'].to_list(), 'Production [kt]', c, axs[0], sf_col_SD='Production|std')
        # yield
        e50_yield_data_analysis.barDfColumn(xdata, xc, 'Yield|mean', xc['adm_name|first'].to_list(),
                                            'Yield [' + yield_unit + ']', c, axs[1], sf_col_SD='Yield|std')
        # area
        if area_unit == 'ha':
            divider = 100
        else:
            print('Measurement units not foreseen')
            exit()
        xc['Area|mean'] = xc['Area|mean'] / divider
        xc['Area|std'] = xc['Area|std'] / divider
        e50_yield_data_analysis.barDfColumn(xdata, xc, 'Area|mean', xc['adm_name|first'].to_list(),
                                            r'${\rm Area \/ [km^2]}$', c, axs[2], sf_col_SD='Area|std')

        # % area by admin
        e50_yield_data_analysis.barDfColumn(xdata, xc, 'Crop_perc_area|', xc['adm_name|first'].to_list(),
                                         '% crop area by admin. unit', c, axs[3], sf_col_SD=None)
        fig.subplots_adjust(bottom=0.25)
        fig.subplots_adjust(left=0.15)
        fig.tight_layout()
        fig.savefig(os.path.join(outDir, config.AOI + '_bar_' + c + str(prct2retain) + '.png'))
        plt.close()






