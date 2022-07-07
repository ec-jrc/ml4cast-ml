import pandas as pd
import numpy as np
import b05_Init
import os
import matplotlib.pyplot as plt

def saveXprctProductionPickle(target):
    project = b05_Init.init(target)
    prct2retain = project['prct2retain']
    # prct2retain is the the percentage to retain
    if prct2retain > 100:
        print('requested percentage gt 100')
        exit()
    '''Compute statistics and save a Pickle of the country stats'''
    pd.set_option('display.max_columns', None)
    desired_width=320
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns',10)

    project = b05_Init.init(target)
    dirStat = project['input_dir'] + '/CountryStats'
    dirOut =  project['output_dir']
    if not os.path.exists(dirOut):
        os.makedirs(dirOut)

    regNames = pd.read_csv(dirStat + '/' + project['AOI'] +'_REGION_id.csv')
    stats = pd.read_csv(dirStat + '/' + project['AOI'] +'_STATS.csv')
    units = pd.read_csv(dirStat + '/' + project['AOI'] + '_measurement_units.csv')
    area_unit = units['Area'].values[0]
    yield_unit = units['Yield'].values[0]
    #set space to nan
    stats = stats.replace(r'^\s+$', np.nan, regex=True)
    stats['Yield'] = pd.to_numeric(stats['Yield'])
    #stats has a lot of 0, likely no data
    stats = stats.replace(0.0, np.nan)
    crop_name = pd.read_csv(dirStat + '/' + project['AOI'] +'_CROP_id.csv')

    stats = pd.merge(stats, regNames, left_on=['Region_ID'], right_on=['AU_code'])
    stats = pd.merge(stats, crop_name, on=['Crop_ID'])
    stats.insert(loc=5, column='Production', value=stats['Area'] * stats['Yield'])
    statsAll = stats.copy()
    #keep only from year of interest for the admin lelevl stats
    stats = stats[stats['Year'] >= project['timeRange'][0]]




    #save a file with missing data (all regions, no matter 90% production or not)
    tmp = stats.copy()
    tmp['Null'] = tmp['Yield'].isnull()
    tmp = tmp[(tmp['Null'] == True) & (tmp['Year'] >= project['timeRange'][0])]
    tmp = tmp.sort_values(by=['Crop_name','AU_code','Year'])
    tmp.to_csv(dirOut + '/' + project['AOI'] + '_missing_all_regions.csv')


    #Mean by: Region, Crop
    x = stats.groupby(['Region_ID', 'Crop_ID']). \
         agg({'ASAP1_ID':'first','AU_name':'first','Crop_name':'first','Production': ['mean', 'std'], 'Yield': ['mean', 'std'], 'Area': ['mean', 'std']})
    x = x.sort_values(by=['Crop_ID', ('Production', 'mean')], ascending=False)
    # add an index 0, 1, ..
    x.reset_index(inplace=True)

    #Compute, by crop
    x[('Cum_sum_production','')] = x.groupby('Crop_ID')[[('Production','mean')]].apply(lambda u: u.cumsum())
    x[('Perc_production','')] = x.groupby('Crop_ID')[[('Production','mean')]].apply(lambda u: u/u.sum()*100)
    x[('Cum_perc_production','')] = x.groupby('Crop_ID')[[('Perc_production','')]].apply(lambda u: u.cumsum())
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

    #x = x[x[('Cum_perc_production', '')] < 90]
    x.to_pickle(dirOut + '/' + project['AOI'] +'_stats'+str(prct2retain)+'.pkl')
    x.to_csv(dirOut + '/' + project['AOI'] +'_stats'+str(prct2retain)+'.csv')


    #save only stats for the 90% regions that will be used
    statsX = statsAll.copy()
    for c in crops:
        tmp = statsX[statsX['Crop_name'] == c]
        regionsX = x[x['Crop_name','first'] == c]['Region_ID'].unique()
        tmp = tmp[tmp['Region_ID'].isin(regionsX)]
        statsX = statsX.drop(statsX[statsX['Crop_name'] == c].index)
        # statsX = statsX.append(tmp)
        statsX = pd.concat([statsX, tmp])

    # save stats that will be used by ML pipeline
    statsX.to_csv(dirOut + '/' + project['AOI'] + '_stats.csv')
    statsX.to_pickle(dirOut + '/' + project['AOI'] + '_stats.pkl')

    # do some basic outlier testing testing
    df = statsX.copy()
    df["Issue"] = ""
    out_df = df[df['Area'] < 0.0]
    df["Issue"] = 'Negative area'
    tmp = df[df['Yield'] < 0.0]
    tmp["Issue"] = 'Negative yield'
    # out_df = out_df.append(tmp)
    out_df = pd.concat([out_df, tmp])

    # keep those that are outside 2 sd
    for c in crops:
        dfc = df.loc[df['Crop_name'] == c]
        tmp = dfc.loc[(np.abs(dfc['Area'] - dfc['Area'].mean()) > (2 * dfc['Area'].std()))].copy()
        tmp["Issue"] = 'Area outside 2 SD'
        out_df = pd.concat([out_df, tmp])
        tmp = dfc.loc[(np.abs(dfc['Yield'] - dfc['Area'].mean()) > (2 * dfc['Area'].std()))].copy()
        tmp["Issue"] = 'Yield outside 2 SD'
        out_df = pd.concat([out_df, tmp])
    out_df.to_csv(dirOut + '/' + project['AOI'] + '_outlier.csv')



    #Save missing records on 90% regions that will be used
    statsX['Null'] = statsX['Yield'].isnull()
    statsX = statsX[(statsX['Null'] == True) & (statsX['Year'] >= project['timeRange'][0])]
    statsX = statsX.sort_values(by=['Crop_name', 'AU_code', 'Year'])
    statsX.to_csv(dirOut + '/' + project['AOI'] + '_missing_in_'+str(prct2retain)+'prct_regions.csv')


    # bar plot of production, area and yield by region of stat90
    crops = x['Crop_name','first'].unique()

    for c in crops:
        # production
        xc = x[x['Crop_name','first'] == c]
        xdata = list(range(len(xc['AU_name'])))
        if area_unit == 'ha' and yield_unit == 't/ha':
            divider = 1000
        elif area_unit == 'ha' and yield_unit == 'kg/ha':
            divider = 1000000
        else:
            print('Measurement units not forseen')
            exit()

        xc['Production', 'mean'] = xc['Production','mean']/divider
        xc['Production', 'std'] = xc['Production','std']/divider
        plt.bar(xdata, xc['Production','mean'].to_list(), yerr=xc['Production','std'].to_list())
        labels = xc['AU_name','first'].to_list()
        labels = [elem[:8] for elem in labels]
        plt.xticks(xdata, labels, rotation='vertical')
        plt.ylabel('Production [kt]')
        plt.title(c + ' (' + str(project['timeRange'][0]) + '-' + str(project['timeRange'][1]) +')')
        plt.subplots_adjust(bottom=0.25)
        plt.subplots_adjust(left=0.15)
        strFn = dirOut + '/' + project['AOI'] + '_' + c + '_' + str(prct2retain)+'prctProd_production.png'
        plt.savefig(strFn.replace(" ", ""))
        plt.close()

        # yield
        plt.bar(xdata, xc['Yield', 'mean'].to_list(), yerr=xc['Yield', 'std'].to_list())
        labels = xc['AU_name', 'first'].to_list()
        labels = [elem[:8] for elem in labels]
        plt.xticks(xdata, labels, rotation='vertical')
        plt.ylabel('Yield [' + yield_unit + ']')
        plt.title(c)
        plt.subplots_adjust(bottom=0.25)
        plt.subplots_adjust(left=0.15)
        strFn = dirOut + '/' + project['AOI'] + '_' + c + str(prct2retain)+'prctProd_yield.png'
        plt.savefig(strFn.replace(" ", ""))
        plt.close()

        # area
        if area_unit == 'ha':
            divider = 100
        else:
            print('Measurement units not forseen')
            exit()
        xc['Area', 'mean'] = xc['Area', 'mean'] / divider
        xc['Area', 'std'] = xc['Area', 'std'] / divider
        plt.bar(xdata, xc['Area', 'mean'].to_list(), yerr=xc['Area', 'std'].to_list())
        labels = xc['AU_name', 'first'].to_list()
        labels = [elem[:8] for elem in labels]
        plt.xticks(xdata, labels, rotation='vertical')
        lbl = r'${\rm Area \/ [km^2]}$'
        plt.ylabel(lbl)
        #plt.ylabel('Area [km2]')
        plt.title(c)
        plt.subplots_adjust(bottom=0.25)
        plt.subplots_adjust(left=0.15)
        strFn = dirOut + '/' + project['AOI'] + '_' + c + str(prct2retain)+'prctProd_area.png'
        plt.savefig(strFn.replace(" ", ""))
        plt.close()


    # bar plot of % area of the crop over the total area of the three by region of stat90
    # compute total area by region using x0 (it has all the regions no matter uf contrib to 90% production)
    x0 = x0.drop('Production', axis = 1, level = 0)
    x0 = x0.drop('Yield', axis=1, level=0)
    x0 = x0.drop('Cum_sum_production', axis=1, level=0)
    x0 = x0.drop('Perc_production', axis=1, level=0)
    x0 = x0.drop('Cum_perc_production', axis=1, level=0)
    x0 = x0.drop('std', axis=1, level=1)
    x0.columns = x0.columns.droplevel(1)
    xTotal = x0.groupby(['Region_ID']).agg({'AU_name': 'first', 'Area': ['sum']})
    xTotal.columns = xTotal.columns.droplevel(1)
    xTotal = xTotal.rename(columns={"Area": "AreaTotCrops"})
    crops = x['Crop_name', 'first'].unique()
    for c in crops:
        xc = x[x['Crop_name', 'first'] == c]
        xc = xc.sort_values(by=[('AU_name', 'first')])
        xdata = list(range(len(xc['AU_name'])))
        xc = pd.merge(xc, xTotal, left_on=[('Region_ID', '')], right_on=['Region_ID'])
        xc['Fraction'] = xc['Area', 'mean'] / xc['AreaTotCrops'] * 100
        plt.bar(xdata, xc['Fraction'].to_list())
        labels = xc['AU_name', 'first'].to_list()
        labels = [elem[:8] for elem in labels]
        plt.xticks(xdata, labels, rotation='vertical')
        plt.ylabel('Region crop area / Total crop area in the region [%]')
        plt.title(c)
        plt.ylim([0, 100])
        plt.subplots_adjust(bottom=0.25)
        plt.subplots_adjust(left=0.15)
        strFn = dirOut + '/' + project['AOI'] + '_' + c + str(prct2retain)+'prctProd_Percent_area_by_au.png'
        plt.savefig(strFn.replace(" ", ""))
        plt.close()
        # plt.show()
    print('end')