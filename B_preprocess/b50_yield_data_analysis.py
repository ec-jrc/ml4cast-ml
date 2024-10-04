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
    outDir = os.path.join(config.data_dir, 'Label_analysis')
    Path(outDir).mkdir(parents=True, exist_ok=True)
    pd.set_option('display.max_columns', None)
    desired_width=10000
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns',100)

    stats = b100_load.LoadLabel(config, save_csv=False, plot_fig=False)
    regNames = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_REGION_id.csv'))
    crop_name = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_CROP_id.csv'))
    units = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_measurement_units.csv'))
    area_unit = units['Area'].values[0]
    yield_unit = units['Yield'].values[0]
    print('Warning: spaces and zeros set to nan in saveYieldStats')
    #set space to nan
    stats = stats.replace(r'^\s+$', np.nan, regex=True)
    stats['Yield'] = pd.to_numeric(stats['Yield'])
    #stats has a lot of 0, likely no data
    stats = stats.replace(0.0, np.nan)

    # stats = pd.merge(stats, regNames, left_on=['adm_id'], right_on=['adm_id'])
    # stats = pd.merge(stats, crop_name, on=['Crop_ID'])
    # stats.insert(loc=5, column='Production', value=stats['Area'] * stats['Yield'])
    # keep only from year of interest for the admin level stats
    stats = stats[stats['Year'] >= config.year_start]

    #save a file with missing data (all regions, no matter 90% production or not)
    tmp = stats.copy()
    tmp['Null'] = tmp['Yield'].isnull()
    tmp = tmp[tmp['Null'] == True]
    tmp = tmp.sort_values(by=['Crop_name','adm_id','Year'])
    if len(tmp.index)>0:
        print('Missing records are present, inspect ' + os.path.join(config.data_dir, config.AOI + '_missing_data.csv'))
    tmp.to_csv(os.path.join(outDir, config.AOI + '_missing_data.csv'), index=False)

    #Mean by: Region, Crop
    x = stats.groupby(['adm_id', 'Crop_ID']). \
         agg({'adm_id':'first','adm_name':'first','Crop_name':'first','Production': ['mean', 'std'], 'Yield': ['mean', 'std', 'count'], 'Area': ['mean', 'std']})
    # sort by production
    x = x.sort_values(by=['Crop_ID', ('Production', 'mean')], ascending=False)
    # add an index 0, 1, ..
    x.reset_index(inplace=True)

    #Compute, by crop

    #by crop in all regions
    x[('Crop_sum_production', '')] = x.groupby('Crop_ID')[[('Production', 'mean')]].transform('sum')
    x[('Cum_sum_production', '')] = x.groupby('Crop_ID')[[('Production', 'mean')]].transform('cumsum')
    x[('Perc_production', '')] = x[('Production','mean')] / x[('Crop_sum_production', '')] * 100
    x[('Cum_perc_production', '')] = x[('Cum_sum_production','')] / x[('Crop_sum_production', '')] * 100

    # by region by crop
    x[('Crop_sum_area', '')] = x.groupby('adm_id')[[('Area', 'mean')]].transform('sum')
    x[('Perc_area', '')] = x[('Area','mean')] / x[('Crop_sum_area', '')] * 100


    # keep a copy with all for later
    x0 = x.copy(deep=True)
    # keep only the largest up to 90% production, by crop
    # 29 04 2021: Mic, at least 90
    crops = x['Crop_name', 'first'].unique()
    for c in crops:
        tmp = x[x['Crop_name','first'] == c]
        tmp = tmp.reset_index(drop=True)
        tmp = tmp.sort_values(by = 'Cum_perc_production')
        if prct2retain != 100:
            ind = tmp[tmp['Cum_perc_production'] >= prct2retain].index[0]
            tmp = tmp.iloc[0:ind + 1]

        x = x.drop(x[x['Crop_name', 'first'] == c].index)
        x = pd.concat([x, tmp])
    # remove multi column
    y = x.copy()
    y.columns = y.columns.map(lambda v: '|'.join([str(i) for i in v]))
    y.to_csv(os.path.join(outDir, config.AOI + '_LTstats_retainPRCT' + str(prct2retain) + '.csv'), index=False)

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
        e50_yield_data_analysis.barDfColumn(xdata, xc, 'Perc_area|', xc['adm_name|first'].to_list(),
                                         '% crop area by admin. unit', c, axs[3], sf_col_SD=None)
        fig.subplots_adjust(bottom=0.25)
        fig.subplots_adjust(left=0.15)
        fig.tight_layout()
        fig.savefig(os.path.join(outDir, config.AOI + '_bar_' + c + str(prct2retain) + '.png'))
        plt.close()
        # print('defguj')





    # # bar plot of % area of the crop over the total area of the three by region of stat90
    # # compute total area by region using x0 (it has all the regions no matter uf contrib to 90% production)
    # x0 = x0.drop('Production', axis = 1, level = 0)
    # x0 = x0.drop('Yield', axis=1, level=0)
    # x0 = x0.drop('Cum_sum_production', axis=1, level=0)
    # x0 = x0.drop('Perc_production', axis=1, level=0)
    # x0 = x0.drop('Cum_perc_production', axis=1, level=0)
    # x0 = x0.drop('std', axis=1, level=1)
    # x0.columns = x0.columns.droplevel(1)
    # xTotal = x0.groupby(['adm_id']).agg({'adm_name': 'first', 'Area': ['sum']})
    # xTotal.columns = xTotal.columns.droplevel(1)
    # xTotal = xTotal.rename(columns={"Area": "AreaTotCrops"})
    # crops = x['Crop_name', 'first'].unique()
    # for c in crops:
    #     xc = x[x['Crop_name', 'first'] == c]
    #     xc = xc.sort_values(by=[('Area', 'mean')], ascending=False)
    #     xdata = list(range(len(xc['adm_name'])))
    #     xc.columns = xc.columns.map(lambda x: '|'.join([str(i) for i in x]))
    #     xc = pd.merge(xc, xTotal, left_on=['adm_id|'], right_index=True)
    #     xc['Fraction'] = xc['Area|mean'] / xc['AreaTotCrops'] * 100
    #     plt.bar(xdata, xc['Fraction'].to_list())
    #     labels = xc['adm_name|first'].to_list()
    #     labels = [elem[:8] for elem in labels]
    #     plt.xticks(xdata, labels, rotation='vertical')
    #     plt.ylabel('Region crop area / Total crop area in the region [%]')
    #     plt.title(c)
    #     plt.ylim([0, 100])
    #     plt.subplots_adjust(bottom=0.25)
    #     plt.subplots_adjust(left=0.15)
    #     plt.savefig(os.path.join(outDir, config.AOI + '_bar_' + c + str(prct2retain) + '_Percent_area_by_au.png'))
    #     plt.close()
