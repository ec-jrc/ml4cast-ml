import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from B_preprocess import b100_load
from E_viz import e50_yield_data_analysis


def saveYieldStats(config, prct2retain=100):
    '''Compute statistics and save csv of the country stats, plot yield correlation among crops
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

    # quality check and outlier removal
    stat_file = os.path.join(config.data_dir, config.AOI + '_STATS.csv')
    stats = b100_load.LoadLabel(stat_file, config.year_start, config.year_end, make_charts=True, perc_threshold=-1, crops_names=config.crops)

    units = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_measurement_units.csv'))
    area_unit = units['Area'].values[0]
    yield_unit = units['Yield'].values[0]
    if 'Production' in units.columns:
        prod_units = units['Production'].values[0]
    else:
        prod_units = 'kt'


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

    # look into correlation among crops [(config.crops])
    yield_cols = [x + '_yield' for x in config.crops]
    cols = ['adm_id', 'Year', 'adm_name'] + yield_cols
    df_corr = pd.DataFrame(columns=cols)
    ids = stats['adm_id'].unique()
    yrs = stats['Year'].unique()

    for y in yrs:
        for i in ids:
            yield_values = []
            for c in config.crops:
                # if we are working on 90% data, one or more crop may be missing (the admin is there for other but missing for one or more)
                row = stats.loc[
                    (stats['adm_id'] == i) & (stats['Year'] == y) & (stats['Crop_name'] == c), 'Yield']  # .iloc[0]
                if row.empty:
                    val = np.nan
                else:
                    val = row.iloc[0]
                yield_values.append(val)
            adm_name = stats.loc[stats['adm_id'] == i, 'adm_name'].iloc[0]
            df_corr.loc[len(df_corr)] = [i, y, adm_name] + yield_values
    # pairwise combinations
    # n = math.comb(len(yield_cols), 2)

    # Prepare year-to-color mapping
    years = sorted(df_corr['Year'].unique())
    year_to_idx = {year: idx for idx, year in enumerate(years)}
    df_corr['Year_idx'] = df_corr['Year'].map(year_to_idx)
    # husl, tab20, tab20b, Set3
    colors = sns.color_palette('tab20', len(years))
    cmap = mcolors.ListedColormap(colors)

    # Set up subplots
    n = len(yield_cols) * (len(yield_cols) - 1) // 2
    if n > 0: #with one crop n = 0, no need of correlation
        fig, axs = plt.subplots(1, n, figsize=(n * 4, 5), constrained_layout=False)
        if n == 1:      #when there are 2 crops, n = 1
            axs = [axs]
        else:
            axs = axs.flatten()
        # axs = axs.flatten()
        c = 0

        for i in range(len(yield_cols)):
            for j in range(i + 1, len(yield_cols)):
                column1 = yield_cols[i]
                column2 = yield_cols[j]

                axs[c].scatter(
                    df_corr[column1],
                    df_corr[column2],
                    c=df_corr['Year_idx'],
                    cmap=cmap,
                    edgecolor='k',
                    linewidth=0.3,
                    s=18,
                    alpha=0.6
                )

                axs[c].set_xlabel(column1)
                axs[c].set_ylabel(column2)
                corr_val = df_corr[[column1, column2]].dropna().corr().iloc[0, 1]
                axs[c].set_title(f'r = {corr_val:.2f}')
                c += 1

        # Create legend on the right
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=str(yr),
                   markerfacecolor=colors[idx], markersize=6, markeredgecolor='k')
            for yr, idx in year_to_idx.items()
        ]
        fig.legend(handles=legend_elements, title="Year", loc='center left', bbox_to_anchor=(0.92, 0.53))

        # Adjust layout for legend space
        plt.tight_layout(rect=[0, 0, 0.90, 1])

        # Save figure
        fig_name = os.path.join(outDir, 'AAA' + config.AOI + '_yield_corr' + str(prct2retain) + '.png')
        fig.savefig(fig_name, dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Last5yrs stats
    if True:
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

        # # keep a copy with all for later
        # x05 = x5.copy(deep=True)
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
        # plot total crop area by crop to understand importance
        grouped = y5.groupby('Crop_name|first')
        sum_area = grouped['Area|mean'].sum()
        mask = y5['Area|mean'].isna() & y5['Yield|mean'].notna()
        # Then, group by 'Crop_name|first' and calculate the percentage
        percentage_yield_covered_by_area = (1 -len(y5[mask].groupby('Crop_name|first')) / len(y5.groupby('Crop_name|first'))) * 100
        # Combine the results
        areaTot5y = pd.DataFrame({
            'Area|mean': sum_area,
            'percentage_yield_covered_by_area': percentage_yield_covered_by_area
        }).reset_index()

        # areaTot5y = y5.groupby('Crop_name|first')['Area|mean'].sum().reset_index()
        areaTot5y = areaTot5y.sort_values(by='Area|mean', ascending=False)
        # Create a bar plot
        plt.figure(figsize=(12, 5))
        plt.bar(areaTot5y['Crop_name|first'].values, areaTot5y['Area|mean'].values)
        for i, (value, top_value) in enumerate(zip(areaTot5y['Area|mean'].values, areaTot5y['percentage_yield_covered_by_area'].values)):
            plt.text(i, value + 1, str(round(top_value)), ha='center')
        plt.title('Values above bars is % of units having area data')
        plt.ylabel('Total area')
        plt.xticks(rotation=90, fontsize=12)  # Rotate x-axis labels for better readability
        plt.tight_layout()
        plt.savefig(os.path.join(outDir,'AAA' + config.AOI + '_5yrsStats_retainPRCT' + str(prct2retain) + '_total_crop_area.png'))
        # for the first 3 most important crops, plot area by admin
        top3 = areaTot5y['Crop_name|first'].values[0:3]
        fig, axs = plt.subplots(3, 1, figsize=(25, 15)) #35
        axs = axs.flatten()
        axc = 0
        for c in top3:
            df = y5[y5['Crop_name|first']==c]
            df = df.sort_values(by='Area|mean', ascending=False)
            xlbs = df['adm_name|first'].str.slice(0, 8).values
            axs[axc].bar(xlbs, df['Area|mean'].values)
            axs[axc].set_ylabel('Total area')
            axs[axc].set_title(c)
            axs[axc].set_xticklabels(axs[axc].get_xticklabels(), rotation=45)
            axs[axc].tick_params(axis='x', labelsize=12)
            axc = axc + 1
            print()
        plt.tight_layout()
        plt.savefig(
            os.path.join(outDir, 'AAA' + config.AOI + '_5yrsStats_retainPRCT' + str(prct2retain) + '_top3crops_area_by_adm.png'))
        plt.close()

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
    stats_prct2retain = stats.iloc[:0,:].copy() # empty df but same columns, same dtypes, and no row
    for c in crops:
        adm_id_2retain_for_crop = y[y['Crop_name|first'] == c]['adm_id|'].unique()
        tmp = stats[(stats['Crop_name'] == c) & (stats['adm_id'].isin(adm_id_2retain_for_crop))]
        stats_prct2retain = pd.concat([stats_prct2retain, tmp])
    cleaned_prct2retain_file = stat_file.replace('.csv', '_cleaned' + str(prct2retain) + '.csv')
    # It may happen that some admin level for which I have yield, do not have ASAP data extracted
    # because there is no AFI (e.g. Somalia). I have to point it out to operator, save a file showing which,
    # and removing them from stats


    if isinstance(config.fn_reference_shape, list):
        # this part replace the original extraction with one with stas id
        # name os.path.join(config.data_dir, config.afi + '_original_before_new_adm_id.csv')
        # make sure I do not mess it up
        # when running it multiple time
        if os.path.isfile(os.path.join(config.data_dir, config.afi + '_original_before_new_adm_id.csv')):
            # it is there, so it was run already, delete the normal one and rename this a s norml
            os.remove(os.path.join(config.data_dir, config.afi + '.csv'))
            os.rename(os.path.join(config.data_dir, config.afi + '_original_before_new_adm_id.csv'), os.path.join(config.data_dir, config.afi + '.csv'))
    dfASAP = pd.read_csv(os.path.join(config.data_dir, config.afi + '.csv'))
    if isinstance(config.fn_reference_shape, list):
        regNames = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_REGION_id.csv'))
        # We are in Marocco-like case, differnt boundaries over time.
        # In this case the shp id "adm_id" does not match the one of stats, that reports the sh id in Adjusted_jrc_id_in_shp
        # Here I change the shp id using regNames and
        # I have to manage that one Adjusted_jrc_id_in_shp may be needed by multiple stats ids
        # Create a lookup table with unique couples of adm_id and Adjusted_jrc_id_in_shp
        dfASAP = dfASAP.rename(columns={'adm_id': 'adm_id_in_shp'})
        lookup_table = regNames[['adm_id', 'Adjusted_jrc_id_in_shp']].drop_duplicates()
        # Create a new DataFrame to store the replicated records
        replicated_df = pd.DataFrame()
        # Iterate over each unique Adjusted_jrc_id_in_shp in the df DataFrame
        for jrc_id in dfASAP['adm_id_in_shp'].unique():
            # Get the adm_id values that match the current jrc_id
            matching_adm_ids = lookup_table.loc[lookup_table['Adjusted_jrc_id_in_shp'] == jrc_id, 'adm_id']
            # If there are matching adm_id values, replicate the records
            if not matching_adm_ids.empty:
                # Get the records in df that match the current jrc_id
                matching_records = dfASAP.loc[dfASAP['adm_id_in_shp'] == jrc_id]
                # Replicate the records for each matching adm_id
                for adm_id in matching_adm_ids:
                    # Create a new DataFrame with the replicated records
                    new_records = matching_records.copy()
                    new_records['adm_id'] =adm_id
                    # Append the new records to the replicated DataFrame
                    replicated_df = pd.concat([replicated_df, new_records])

        dfASAP = replicated_df
        shutil.copyfile(os.path.join(config.data_dir, config.afi + '.csv'), os.path.join(config.data_dir, config.afi + '_original_before_new_adm_id.csv'))
        # dfASAP['adm_id'] = dfASAP['adm_id'].astype(str)
        if os.path.isfile(os.path.join(config.data_dir, config.afi + '.csv')):
            # it is there, so it was run already, delete the normal one and rename this a s norml
            os.remove(os.path.join(config.data_dir, config.afi + '.csv'))
        dfASAP.to_csv(os.path.join(config.data_dir, config.afi + '.csv'), index=False) #, mode='w') #, header=True)

    A = set(dfASAP['adm_id'].unique())  # ASAP extraction admin Ids
    B = set(stats_prct2retain['adm_id'].unique())
    doRemove = False
    if A != B:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('b50 Warning')
        # Find elements in A that are not in B
        in_A_not_B = A - B
        # Find elements in B that are not in A
        in_B_not_A = B - A
        if in_A_not_B:
            print("Elements in Asap extractions that are not in Region names:", in_A_not_B)
            print("This may be due to the fact that there is no yield data for these units")
        if in_B_not_A:
            print("Elements in yield stats that are not in Asap extraction:", in_B_not_A)
            doRemove = True
            # sys.exit()
        print('!!!!!!!!!!!!!!!!!!!!!!!!!')
    if doRemove:
        in_B_not_A_list = list(in_B_not_A)
        name_list = [stats_prct2retain[stats_prct2retain['adm_id'] == x]['adm_name'].iloc[0] for x in in_B_not_A_list]
        crops_list_of_list = [list(stats_prct2retain[stats_prct2retain['adm_id'] == x]['Crop_name'].unique()) for x in in_B_not_A_list]
        print('The following admins will be removed from cleaned stats file')
        for xt, yt, zt in zip(in_B_not_A_list, name_list, crops_list_of_list):
            print(xt, yt, zt)
        # print(str(in_B_not_A_list))
        # print(str(name_list))
        pro = input('Type Y to proceed\n')
        if pro == 'Y':
            # print a file
            df_out = pd.DataFrame({"Dropped_admin_with_no_ASAP_data": in_B_not_A_list, "Dropped_admin_name": name_list,
                                   "crops": crops_list_of_list})
            dropped_adm_file = stat_file.replace('.csv', '_adm_dropped' + str(prct2retain) + '.csv')
            df_out.to_csv(dropped_adm_file, index=False)
            stats_prct2retain = stats_prct2retain[~stats_prct2retain["adm_id"].isin(in_B_not_A_list)]
            # remove unmatched records from stas
        else:
            sys.exit('b50 terminated by user')
    stats_prct2retain.to_csv(cleaned_prct2retain_file, index=False)
    # In the case of multipolygon (e.g. Morocco)
    # extract which are the last valid admins on which we would do ope forecasts
    if isinstance(config.fn_reference_shape, list):
        if config.AOI == 'MA':
            #This processing is country specific, here for Morocco
            #having MA2016
            u = stats_prct2retain[stats_prct2retain['fnid'].str.contains('^MA2016', regex=True, na=False)]
            u = u.sort_values(by='Year').drop_duplicates(subset='adm_id', keep='last')
            # u.to_csv(os.path.join(config.data_dir, config.AOI + '_ACTIVE_ADM_MA2016.csv'), index=False)
            #having one valid data after 2015
            filtered_adm_ids = stats_prct2retain[(stats_prct2retain['Year'] > 2015) & (stats_prct2retain['Yield'].notna())]['adm_id'].unique()
            # Use these adm_ids to filter the original DataFrame
            v = stats_prct2retain[stats_prct2retain['adm_id'].isin(filtered_adm_ids)]
            v = v.sort_values(by='Year').drop_duplicates(subset='adm_id', keep='last')
            if not u.equals(v):
                print('b50 u and v are different, check why')
                exit()
            v.to_csv(os.path.join(config.data_dir, config.AOI + '_ACTIVE_ADMINS.csv'), index=False)
        else:
            print('b50, trying to find active admins, but no rule for this AOI, please add')
            exit()


    # bar plot of production, area and yield by region of retained statistics
    crops = x['Crop_name','first'].unique()
    for c in crops:
        # if c == 'Coffee':
        #     print(c)
        # production
        xc = x[x['Crop_name','first'] == c]
        xc = xc.sort_values(by=[('Area', 'mean')], ascending=False)
        xdata = list(range(len(xc['adm_name'])))
        if area_unit == 'ha' and yield_unit == 't/ha':
            divider = 1000
        elif area_unit == 'ha' and yield_unit == 'kg/ha':
            divider = 1000000
        elif prod_units == '100t':
            divider = 10
        else:
            print('Measurement units not foreseen')
            exit()

        xc['Production', 'mean'] = xc['Production','mean']/divider
        xc['Production', 'std'] = xc['Production','std']/divider
        xc.columns = xc.columns.map(lambda v: '|'.join([str(i) for i in v]))
        w = len(xc['adm_name|first'].to_list()) #width based on number of admins
        if w <= 15: # ZA
            w = 10
            h = 10
            thicklblsz = 10
            titlesz = 15
            axisTitlesz = 10
        elif w > 15 and w < 40: # BE
            h = 15  # increase h
            thicklblsz = 30
            titlesz = 40
            axisTitlesz = 30
        elif w >= 40: # ZM
            h = 30 #increase h
            thicklblsz = 30
            titlesz = 40
            axisTitlesz = 30
        fig, axs = plt.subplots(2, 2, figsize=(w, h))
        axs = axs.flatten()
        # plot production
        e50_yield_data_analysis.barDfColumn(xdata, xc, 'Production|mean', xc['adm_name|first'].to_list(), 'Production [kt]', c, axs[0], sf_col_SD='Production|std')
        # yield
        e50_yield_data_analysis.barDfColumn(xdata, xc, 'Yield|mean', xc['adm_name|first'].to_list(),
                                            'Yield [' + yield_unit + ']', c, axs[1], sf_col_SD='Yield|std')
        # area
        if area_unit == 'ha':
            divider = 100
        elif area_unit == 'km2':
            divider = 1
        else:
            print('Measurement units not foreseen')
            exit()
        xc['Area|mean'] = xc['Area|mean'] / divider
        xc['Area|std'] = xc['Area|std'] / divider
        e50_yield_data_analysis.barDfColumn(xdata, xc, 'Area|mean', xc['adm_name|first'].to_list(),
                                            r'${\rm Area \/ [km^2]}$', c, axs[2], sf_col_SD='Area|std')

        # % area by admin
        e50_yield_data_analysis.barDfColumn(xdata, xc, 'Crop_perc_area|', xc['adm_name|first'].to_list(),
                                         '% of national crop area [%]', c, axs[3], sf_col_SD=None)
        for i in [0,1,2,3]:
            axs[i].tick_params(axis='both', which='major', labelsize=thicklblsz)
            axs[i].title.set_size(titlesz)
            axs[i].yaxis.label.set_size(axisTitlesz)

        fig.subplots_adjust(bottom=0.25)
        fig.subplots_adjust(left=0.15)
        fig.tight_layout()
        fig.savefig(os.path.join(outDir, config.AOI + '_bar_' + c + str(prct2retain) + '.png'))
        plt.close()






