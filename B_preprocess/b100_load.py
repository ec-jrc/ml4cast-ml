import sys

import pandas as pd
import numpy as np
from scipy import stats
import pymannkendall as mk
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker




def LoadPredictors_Save_Csv(config, runType):
    ''' Import ASAP data in the csv format,
    check agreement with yiled stats data (regions missing in ASAP or missing in stats)
    polish it, organize it as formatted dekadak data,
    save as csv (.._predictors.csv) '''
    desired_width = 320
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns', 1000)

    dirIn = config.data_dir
    dirOut = config.models_dir

    if runType == 'opeForecast':
        dirInOpe = config.ope_data_dir
        dirOut = config.ope_run_dir
        Path(dirOut).mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(os.path.join(dirInOpe, config.afi + '.csv'))
    else:
        df = pd.read_csv(os.path.join(dirIn, config.afi + '.csv'))

    # General part
    df = df[df['class_name']=='crop']
    df = df[df['classset_name'] == 'static masks']
    # read stats cleaned to check what regions are present
    stat_file = os.path.join(config.data_dir, config.AOI + '_STATS_cleaned' + str(config.prct2retain) + '.csv')
    statsDf = pd.read_csv(stat_file)
    # read the table with id and region name
    regNames = pd.read_csv(os.path.join(dirIn, config.AOI + '_REGION_id.csv'))

    A = set(df['adm_id'].unique())       # ASAP extraction admin Ids
    B = set(statsDf['adm_id'].unique())
    if A != B:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!')
        # Find elements in A that are not in B
        in_A_not_B = A - B
        # Find elements in B that are not in A
        in_B_not_A = B - A
        if in_A_not_B:
            print("Elements in Asap extractions that are not in stat file:", in_A_not_B)
            print("This may be due to the fact that there is no yield data for these units")
        if in_B_not_A:
            print("Elements in stat file that are not in Asap extraction:", in_B_not_A)
            print('Make sure you run yield stats analysis to exclude the latter list')
        print('Stat file: ' + stat_file)
        print('ASAP file: ' + os.path.join(dirIn, config.afi + '.csv'))
        print('!!!!!!!!!!!!!!!!!!!!!!!!!')

    # now link it with adm_id and name
    df = pd.merge(df, regNames, left_on=['adm_id'], right_on=['adm_id'])
    # remove the admin name form extraction, not needed, use the one from region file
    df = df.rename(columns={'adm_name_x': 'adm_name_extraction', 'adm_name_y': 'adm_name'})
    df.drop('adm_name_extraction', axis=1, inplace=True)
    # get first NDVI or FPAR time and drop everything before
    minDate = df[(df['variable_name'] == 'NDVI') | (df['variable_name'] == 'FPAR')]['date'].min()
    df = df[df['date'] >= minDate]
    # fidf date, add a column with date
    df['Date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    # remove useless stuff
    df = df.drop(['var_id','classset_name','classesset_id','class_name','class_id','date'], axis=1)
    # add dekad of the year
    #df['dek'] = df['Date'].map(f_dek_utilities.f_datetime2dek)
    # as of today (2024-07-23, SM no yet gap filled), SM may have missing data in 2001 and 2002, use linear interpolation to fix
    # Now (2024 09 13), SM is nicely gapfilled using valencia method, this following piece of code
    # is kept to run old versions, can be dismissed once th testing is finished
    for au in df['adm_id'].unique():
        df.loc[(df['adm_id'] == au) & (df['variable_name'] == 'soil_moisture'), 'mean'] =  df.loc[(df['adm_id'] == au) & (df['variable_name'] == 'soil_moisture'), 'mean'].interpolate(method='linear', axis=0)
    # save files
    df.to_csv(os.path.join(dirOut, config.AOI + '_predictors.csv'), index=False)

def build_features(config, runType):
    # created monthly features and then reashape as scikit format
    # config is the config object
    pd.set_option('display.max_columns', None)
    desired_width = 320
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns', 10)

    dirOut = config.models_dir
    if runType == 'opeForecast':
        dirOut = config.ope_run_dir

    sosMonth = config.sosMonth
    eosMonth = config.eosMonth
    # open predictors
    fn = os.path.join(dirOut, config.AOI + '_predictors.csv')
    df = pd.read_csv(fn)
    if config.useSF == True:
        # check SF is there
        if df['variable_name'].str.startswith('SF').sum() < 10:
            raise ValueError("Use of Seasonal Forecast required, but no SF in _predictors.csv")
        # A given month MM contains Seasonal Forecast means of that month MM with date MM/01,
        # when preparing the data (load), this has to be assigned to the previous month MM-1/01
        # to be avaiable in tuning and forecasting
        # shift - 1 month
        df['Date'] = pd.to_datetime(df['Date'])
        df.loc[df['variable_name'].str.contains('SF_'), 'Date'] = (df['Date'] + pd.DateOffset(months=-1)).dt.strftime('%Y-%m-%d')
        # # check: df_month[(df_month['adm_id']==2223) & (df_month['Year']==2000) & ((df_month['variable_name']=='SF_t_1') | (df_month['variable_name']=='temperature'))]


    df["Datetime"] = pd.to_datetime(df.Date)
    df["Year"] = df.Datetime.dt.year
    df["Month"] = df.Datetime.dt.month
    # Aggregate to monthly values (min, max, avg, sum for variable values)
    df_month = df.groupby(by=[df.Year, df.Month,
                              df.variable_name, df.adm_name]).agg(
                                # adm_name=('adm_name', 'first'),
                                adm_id=('adm_id', 'first'),
                                Date=('Date', 'min'), mean=('mean', 'mean'),  # Year=('Year','first'),Month=('Month','first'),
                                min=('mean', 'min'), max=('mean', 'max'), sum=('mean', 'sum'))
    df_month = df_month.reset_index()
    # df_month = df_month.drop(columns=['adm_name'])
    if sosMonth < eosMonth:
        months = list(range(sosMonth, eosMonth+1))
    else:
        months = list(range(sosMonth, 12 + 1)) + list(range(1, eosMonth + 1))
    month_index = list(range(1, len(months)+1))
    df_month = df_month.loc[df_month['Month'].isin(months)]
    di = dict(zip(months, month_index))
    df_month['Month_index'] = df_month['Month'].map(di)
    df_month.insert(2, "Month_index", df_month.pop("Month_index"))
    # add year of eos
    if sosMonth < eosMonth:
        df_month['YearOfEOS'] = df_month['Year']
    else:
        df_month['YearOfEOS'] = df_month['Year']
        # df_month.loc[(df_month['Month'] <= 12) & (df_month['Month'] > eosMonth), 'YearOfEOS'] = df_month['YearOfEOS'] + 1
        df_month.loc[df_month['Month'] > eosMonth, 'YearOfEOS'] = df_month['YearOfEOS'] + 1
    df_month.insert(2, "YearOfEOS", df_month.pop("YearOfEOS"))

    # At the beginning of the time series, keep only full series (MODIS starts in 2000 05 21), so month 5 is the first
    # a season going from month 1 to 7 is incomplete for YearOfEOS 2000
    # a seson going from dek 4 year Y to dek 2 year Y+1 is incomplete for YearOfEOS 2000 and 2001
    # ->this first two year must be checked
    YearOfEOS2check = df_month["YearOfEOS"].sort_values().unique()[0:2]
    for year2check in YearOfEOS2check:
        month_list = df_month[df_month["YearOfEOS"] == year2check]["Month"].unique()
        if len(month_list) != len(months):
            df_month = df_month[df_month["YearOfEOS"] != year2check]

    df_month.to_csv(os.path.join(dirOut, config.AOI + '_monthly_features.csv'), index=False)



    #Reshape the df to scikit format
    MonthSep = 'M'
    ivars = config.ivars
    ivars_short = config.ivars_short
    AU_list = df_month['adm_id'].unique()
    init = 0
    for au in AU_list:
        df_au = df_month[df_month['adm_id'] == au]
        # dfM_au = k[k['adm_id'] == au]
        YY_list = df_au['YearOfEOS'].unique()
        for yy in YY_list:
            df_au_yy =  df_au[df_au['YearOfEOS'] == yy]
            row = [df_au['adm_id'].iloc[0],df_au['adm_name'].iloc[0], df_au_yy['YearOfEOS'].iloc[0]]
            columns = ['adm_id', 'adm_name', 'YearOfEOS']
            for v, vs in zip(ivars, ivars_short):
                df_au_yy_v = df_au_yy[df_au_yy['variable_name'] == v]
                if len(df_au_yy_v) == 0:
                    print('b100_load.build_features, no data for variable', v, 'in year', yy, 'au', au )
                mm_list = np.sort(df_au_yy['Month_index'].unique())
                for mm in mm_list:
                    # now is one variable during a pheno phase of a year of an AU (all info in the list)
                    dfM_au_yy_v_mm = df_au_yy_v[df_au_yy_v['Month_index'] == mm]
                    if len(dfM_au_yy_v_mm) > 0: # one (SM) may not be available
                        if v == 'rainfall':
                            row.append(dfM_au_yy_v_mm['sum'].iloc[0])
                            columns.append(vs + 'Sum' + MonthSep + str(mm))
                        else:
                            row.append(dfM_au_yy_v_mm['mean'].iloc[0])
                            columns.append(vs + MonthSep + str(mm))
                        if (v == 'temperature') or (v == 'NDVI') or (v == 'FPAR'):
                            columns.append(vs + 'min' + MonthSep + str(mm))
                            row.append(dfM_au_yy_v_mm['min'].iloc[0])
                        if not("SF_" in v) and ((v == 'temperature') or (v == 'NDVI') or (v == 'FPAR')):
                            columns.append(vs + 'max' + MonthSep + str(mm))
                            row.append(dfM_au_yy_v_mm['max'].iloc[0])
            if init == 0:
                df = pd.DataFrame([row], columns=columns)
                init = 1
            else:
                df2 = pd.DataFrame([row], columns=columns)
                df = pd.concat([df, df2])
    df.to_csv(os.path.join(dirOut, config.AOI + '_features4scikit.csv'), index=False)

    # df.to_pickle(dirOut + '/' + project['AOI'] + '_pheno_features4scikit.pkl')





def LoadLabel_check_quality_and_clean(stat_file, start_year, end_year, make_charts=False, perc_threshold=-1, crops_names=None):
    '''
    This function is loading stats (without excluding admin with missing values)
    and making quality checks. It saves a cleaned version of the stats
    crops_names: pass a list of crops if tou don't want all the crops in stats to be cleaned (e.g. Harvest data have minor crops that are not of interest
    perc_threshold: z-score >zScoreTreshold or <-zScoreTreshold will be flagged if the abd difference with respect to mean is >  perc_threshold, set to -1 (default) to omit this check
    '''
    zScoreTreshold = 3 # theshold for finding ouliers
    df = pd.read_csv(stat_file)
    units = pd.unique(df['adm_id'])

    if crops_names is None:
        crops_names = pd.unique(df['Crop_name'])

    units_names = pd.unique(df['adm_name'])
    # crops_names = pd.unique(df['Crop_name'])
    if len(units) != len(units_names):
        print('!!!!! b100 LoadLabel: the number of admin id is not equal the number of admin names')
        print('N unit ids: ' + str(len(units)))
        print('N admin names: ' + str(len(units_names)))
        print('Print adm ids that have multiple names')
        for id in units:
            names = df[df['adm_id'] == id]['adm_name'].unique()
            if len(names) > 1:
                print(id, names)
        print('This may be due to multiple yield data units assigned to the same shp polygon (e.g. Morocco)')

    # Initialize columns for 'Outlier', 'Duplicate', and 'LowYield' with dtype 'object'
    df['Outlier'] = pd.Series(dtype='object')
    df['Duplicate'] = pd.Series(dtype='object')
    df['LowYield'] = pd.Series(dtype='object')

    # summary = pd.DataFrame(columns=[
    #     'adm_id', 'adm_name', 'Crop_ID', 'Crop_name',
    #     'Outliers values', 'Outliers years', 'Outlier types',
    #     'Duplicates n', 'Duplicates years', 'Trend'
    # ])
    summary = pd.DataFrame()

    for unit, unit_name in zip(units, units_names):
        data_unit = df[df['adm_id'] == unit]

        for crop_name in crops_names:
            data_crop = data_unit[data_unit['Crop_name'] == crop_name]
            duplicate_rows = data_crop[data_crop['Year'].isin(data_crop['Year'][data_crop['Year'].duplicated()])]
            # Print the duplicate rows if they exist
            if not duplicate_rows.empty:
                print("Duplicate rows found:")
                print(duplicate_rows)

            if len(data_crop) > 0:
                years_duplicate = []
                duplicate_values = []
                outlier_values = []
                outlier_years = []
                outlier_types = []
                trend_type = []

                # Ensure data is sorted by Year
                data_crop = data_crop.sort_values(by=['Year'])

                # Filter for the specified year range for outliers only
                data_year = data_crop[(data_crop['Year'] >= start_year) & (data_crop['Year'] <= end_year)]

                # Positive and negative outliers
                positive_outliers = data_year[stats.zscore(data_year['Yield'], nan_policy='omit') > zScoreTreshold]
                negative_outliers = data_year[stats.zscore(data_year['Yield'], nan_policy='omit') < -zScoreTreshold]
                avg = np.nanmean(data_year['Yield'].values)

                for outlier_df, flag in zip([positive_outliers, negative_outliers], ['Removed', 'Flagged']):
                    percent_diff = abs((outlier_df['Yield'].values - avg) / avg) * 100
                    for idx, pd_diff in enumerate(percent_diff):
                        if pd_diff >= perc_threshold:
                            outlier_row = outlier_df.iloc[idx]
                            df_idx = outlier_row.name
                            df.loc[df_idx, 'Outlier'] = flag
                            outlier_values.append(outlier_row['Yield'])
                            outlier_years.append(outlier_row['Year'])
                            outlier_types.append(flag)

                # Check for duplicates across the full dataset
                duplicate_rows = data_crop[data_crop['Yield'].shift() == data_crop['Yield']]
                if not duplicate_rows.empty:
                    years_duplicate = duplicate_rows['Year'].tolist()
                    duplicate_values = duplicate_rows['Yield'].tolist()
                    df.loc[duplicate_rows.index, 'Duplicate'] = 'Removed'

                # Trend detection
                tmp_trend = 'not enough samples'
                trend_detected = False  # Flag to check if a trend is detected
                if len(data_year) > 1:
                    try:
                        mk_test = mk.original_test(data_year['Yield'])
                        tmp_trend = mk_test.trend
                        if tmp_trend in ['increasing', 'decreasing']:
                            trend_detected = True
                    except ZeroDivisionError:
                        tmp_trend = 'not enough samples'
                trend_type.append(tmp_trend)

                # Add results to summary
                summary = pd.concat([summary, pd.DataFrame([{
                    'adm_id': unit,
                    'adm_name': unit_name,
                    # 'Crop_ID': crop,
                    'Crop_name': crop_name,
                    'Outliers values': '|'.join(map(str, outlier_values)),
                    'Outliers years': '|'.join(map(str, outlier_years)),
                    'Outlier types': '|'.join(outlier_types),
                    'Duplicates n': len(duplicate_rows),
                    'Duplicates years': '|'.join(map(str, years_duplicate)),
                    'Trend': '|'.join(trend_type),
                }])], ignore_index=True)

                # Generate plots if requested
                if make_charts:
                    graph_df = data_crop[['Year', 'Yield']].sort_values(by='Year')
                    plt.plot(graph_df['Year'], graph_df['Yield'], label='Yield', linewidth=1.5, color='#1f77b4')

                    # Plot trend line only if a trend was detected
                    if trend_detected and len(data_year['Year'].unique()) > 1 and data_year['Yield'].var() > 0:
                        try:
                            z = np.polyfit(data_year['Year'], data_year['Yield'], 1)
                            p = np.poly1d(z)
                            plt.plot(data_year['Year'], p(data_year['Year']), linestyle='--', color='green',
                                     label='Trend')
                        except np.linalg.LinAlgError:
                            pass

                    # Mark duplicates and separate outliers as Removed (red) and Flagged (green)
                    removed_outliers = df[
                        (df['Outlier'] == 'Removed') & (df['adm_id'] == unit) & (df['Crop_name'] == crop_name)]
                    flagged_outliers = df[
                        (df['Outlier'] == 'Flagged') & (df['adm_id'] == unit) & (df['Crop_name'] == crop_name)]

                    # Plot outliers as red and green circles
                    plt.scatter(removed_outliers['Year'], removed_outliers['Yield'],
                                color='red', marker='o', s=100, label='Outliers Removed', facecolors='none')
                    plt.scatter(flagged_outliers['Year'], flagged_outliers['Yield'],
                                color='green', marker='o', s=100, label='Outliers Flagged', facecolors='none')

                    # Plot duplicates
                    plt.scatter(years_duplicate, duplicate_values,
                                color='orange', marker='x', s=100, label='Duplicates')

                    # Add vertical lines for start and end year
                    plt.axvline(x=start_year, color='red', linestyle=':', label='Start Year')
                    plt.axvline(x=end_year, color='red', linestyle='--', label='End Year')

                    # Ensure consistent legend
                    custom_handles = {
                        'Yield': plt.Line2D([], [], color='#1f77b4', linewidth=1.5),
                        'Trend': plt.Line2D([], [], linestyle='--', color='green'),
                        'Outliers Removed': plt.Line2D([], [], color='red', marker='o', linestyle='None', markersize=10,
                                                       markerfacecolor='none'),
                        'Outliers Flagged': plt.Line2D([], [], color='green', marker='o', linestyle='None',
                                                       markersize=10, markerfacecolor='none'),
                        'Duplicates Removed': plt.Line2D([], [], color='orange', marker='x', linestyle='None',
                                                         markersize=10),
                        'Start Year': plt.Line2D([], [], color='red', linestyle=':', linewidth=1),
                        'End Year': plt.Line2D([], [], color='red', linestyle='--', linewidth=1)
                    }

                    # Update labels and handles
                    final_handles = [custom_handles[label] for label in
                                     ['Yield', 'Trend', 'Outliers Removed', 'Outliers Flagged', 'Duplicates Removed',
                                      'Start Year', 'End Year']]
                    final_labels = ['Yield', 'Trend', 'Outliers Removed', 'Outliers Flagged', 'Duplicates Removed',
                                    'Start Year', 'End Year']
                    # ste x grid
                    ax = plt.gca()
                    ax.grid(which='major', axis='x', linestyle='--')
                    # Move legend outside the plot area
                    plt.legend(handles=final_handles, labels=final_labels, loc='upper center', fontsize='small',
                               bbox_to_anchor=(1.17, 1.0))

                    # Add titles and labels
                    graph_name = Path(stat_file).stem + ' ' + str(unit_name) + ' ' + str(unit) + ' ' + str(crop_name)
                    plt.title(graph_name)
                    plt.xlabel('Year')
                    plt.ylabel('Yield')

                    # Ensure x-axis tick marks are integers
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                    plt.xticks(fontsize='small')
                    plt.yticks(fontsize='small')

                    # Save the chart
                    graphics_folder = Path(stat_file).parent / 'QC_graphics'
                    graphics_folder.mkdir(parents=True, exist_ok=True)
                    out_png = graphics_folder / (graph_name.replace(" ", "_") + '.png')

                    plt.savefig(out_png, bbox_inches='tight')
                    plt.close()

    # Determine rows to flag as 'LowYield'
    valid_area_prod = df['Area'].notna() & (df['Area'] > 0) & df['Production'].notna() & (df['Production'] > 0)
    ratio = df.loc[valid_area_prod, 'Production'] / df.loc[valid_area_prod, 'Area']
    mask_low_yield = (df['Yield'] == 0) & valid_area_prod & (ratio >= 0.5)
    df.loc[mask_low_yield, 'LowYield'] = 'Removed'

    # Save the flagged file before cleaning
    flagged_file = stat_file.replace('.csv', '_flagged.csv')
    df.to_csv(flagged_file, index=False)

    # CLEANING SECTION: Drop rows based on flags and null values
    cleaned_df = df[
        ~(df['Outlier'] == 'Removed') &
        ~(df['Duplicate'] == 'Removed') &
        ~(df['LowYield'] == 'Removed') &
        df['Area'].notna() & df['Production'].notna() & df['Yield'].notna()
        ].copy()

    # Drop the Outlier, Duplicate, and LowYield columns after cleaning
    cleaned_df = cleaned_df.drop(columns=['Outlier', 'Duplicate', 'LowYield'])

    # Save the cleaned CSV
    cleaned_file = stat_file.replace('.csv', '_cleaned.csv')
    cleaned_df.to_csv(cleaned_file, index=False)

    # Save the summary file
    summary_file = stat_file.replace('.csv', '_summary.csv')
    summary.to_csv(summary_file, index=False)

    # here return the cleaned stats
    return cleaned_df
