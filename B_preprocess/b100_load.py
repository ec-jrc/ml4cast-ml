import pandas as pd
import numpy as np
import os
import datetime
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt



def LoadPredictors_Save_Csv(config, runType):
    ''' Import ASAP data in the csv format, save csv '''
    pd.set_option('display.max_columns', None)
    desired_width=320
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns',1000)

    dirIn = config.data_dir
    dirOut = config.models_dir

    if runType == 'opeForecast':
        dirInOpe = config.ope_data_dir
        dirOut = config.ope_run_dir
        Path(dirOut).mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(os.path.join(dirInOpe, config.AOI + '_ASAP_data.csv'))
    else:
        df = pd.read_csv(os.path.join(dirIn, config.AOI + '_ASAP_data.csv'))

    # General part
    df = df[df['class_name']=='crop']
    df = df[df['classset_name'] == 'static masks']
    # read the table with id and wilaya name
    regNames = pd.read_csv(os.path.join(dirIn, config.AOI + '_REGION_id.csv'))
    # now link it with AU_code and name
    df = pd.merge(df, regNames, left_on=['reg0_id'], right_on=['ASAP1_ID'])
    # get first NDVI time and drop everything before
    minDate = df[(df['variable_name'] == 'NDVI') | (df['variable_name'] == 'FPAR')]['date'].min()
    df = df[df['date'] >= minDate]
    # fidf date, add a column with date
    df['Date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    # remove useless stuff
    df = df.drop(['var_id','classset_name','classesset_id','class_name','class_id','date'], axis=1)
    # add dekad of the year
    #df['dek'] = df['Date'].map(f_dek_utilities.f_datetime2dek)
    # save files
    df.to_csv(os.path.join(dirOut, config.AOI + '_predictors.csv'), index=False)

def build_features(config, runType):
    # config is the config object
    pd.set_option('display.max_columns', None)
    desired_width = 320
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns', 10)

    dirOut = config.models_dir
    if runType == 'opeForecast':
        dirOut = config.ope_run_dir

    sosDek = config.sos
    eosDek = config.eos
    sosMonth = int(np.ceil(sosDek/3))
    eosMonth = int(np.ceil(eosDek/3))
    # open predictors
    fn = os.path.join(dirOut, config.AOI + '_predictors.csv')
    df = pd.read_csv(fn)
    df["Datetime"] = pd.to_datetime(df.Date)
    df["Year"] = df.Datetime.dt.year
    df["Month"] = df.Datetime.dt.month
    df_month = df.groupby(by=[df.Year, df.Month,
                              df.variable_name, df.reg0_name]).agg(
                                AU_name=('AU_name', 'first'), AU_code=('AU_code', 'first'),
                                ASAP1_ID=('ASAP1_ID', 'first'),
                                Date=('Date', 'first'), mean=('mean', 'mean'),  # Year=('Year','first'),Month=('Month','first'),
                                min=('mean', 'min'), max=('mean', 'max'), sum=('mean', 'sum'))
    df_month = df_month.reset_index()
    if sosMonth < eosMonth:
        months = list(range(sosMonth, eosMonth+1))
    else:
        months = list(range(sosMonth, 12 + 1)) + list(range(1, eosMonth + 1))
    month_index = list(range(1, len(months)+1))
    #remove months that are not used
    df_month = df_month.loc[df_month['Month'].isin(months)]
    df_month['First'] = df_month.groupby(['Year'])['Month'].transform('min')
    df_month['Last'] = df_month.groupby(['Year'])['Month'].transform('max')
    # keep only complete years
    if runType == 'tuning':
        df_month = df_month.loc[(df_month['First'] <= sosMonth) & (df_month['Last'] >= eosMonth)]
    df_month = df_month.drop(columns=['First', 'Last','reg0_name'])
    di = dict(zip(months, month_index))
    df_month['Month_index'] = df_month['Month'].map(di)
    df_month.insert(2, "Month_index", df_month.pop("Month_index"))
    # add year of eos
    if sosMonth < eosMonth:
        df_month['YearOfEOS'] = df_month['Year']
    else:
        df_month['YearOfEOS'] = df_month['Year']
        df_month.loc[(df_month['Month'] <= 12) & (df_month['Month'] > eosMonth), 'YearOfEOS'] = df_month['YearOfEOS'] + 1


    df_month.insert(2, "YearOfEOS", df_month.pop("YearOfEOS"))
    df_month.to_csv(os.path.join(dirOut, config.AOI + '_monthly_features.csv'), index=False)


    #Reshape the df to scikit format

    MonthSep = 'M'
    ivars = config.ivars
    ivars_short = config.ivars_short
    AU_list = df_month['AU_code'].unique()
    init = 0
    for au in AU_list:
        df_au = df_month[df_month['AU_code'] == au]
        # dfM_au = k[k['AU_code'] == au]
        YY_list = df_au['YearOfEOS'].unique()
        for yy in YY_list:
            df_au_yy =  df_au[df_au['YearOfEOS'] == yy]
            row = [au,df_au['ASAP1_ID'].iloc[0],df_au['AU_name'].iloc[0], df_au_yy['YearOfEOS'].iloc[0]]
            columns = ['AU_code', 'ASAP1_ID', 'AU_name', 'YearOfEOS']
            for v, vs in zip(ivars, ivars_short):
                df_au_yy_v = df_au_yy[df_au_yy['variable_name'] == v]
                if len(df_au_yy_v) == 0:
                    print('b100_load, no data for variable', v, 'in year', yy)
                mm_list = np.sort(df_au_yy['Month_index'].unique())
                for mm in mm_list:
                    # now is one variable during a pheno phase of a year of an AU (all info in the list)
                    dfM_au_yy_v_mm = df_au_yy_v[df_au_yy_v['Month_index'] == mm]
                    if v == 'rainfall':
                        row.append(dfM_au_yy_v_mm['sum'].iloc[0])
                        columns.append(vs + 'Sum' + MonthSep + str(mm))
                    else:
                        row.append(dfM_au_yy_v_mm['mean'].iloc[0])
                        columns.append(vs + MonthSep + str(mm))
                    if (v == 'temperature') or (v == 'NDVI'):
                        columns.append(vs + 'min' + MonthSep + str(mm))
                        row.append(dfM_au_yy_v_mm['min'].iloc[0])
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


def LoadLabel_Exclude_Missing(config, save_csv = True, plot_fig= False, verbose= True):
    '''
    This function is just checking for NaN in stats
    For stats data, remove regions-crop with missing data
    '''

    # get yield stats
    stats = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_stats.csv'))
    #'AU_name' may not be present
    if ('AU_name' in stats.columns) == False:
        names = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_REGION_id.csv'))
        stats = pd.merge(stats, names, how='left', left_on=['Region_ID'], right_on=['AU_code'])
    if ('Crop_name' in stats.columns) == False:
        names = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_CROP_id.csv'))
        stats = pd.merge(stats, names, how='left', left_on=['Crop_ID'], right_on=['Crop_ID'])
    # drop years before period of interest
    stats = stats.drop(stats[stats['Year'] < config.year_start].index).sort_values('AU_name')


    crops = stats['Crop_ID'].unique()
    if verbose:
        print('Nan in yield')
        print('N missing year by region')
    for c in crops:
        statsC = stats[stats['Crop_ID'] == c]
        tmp = statsC[statsC['Yield'].isnull()]
        if len(tmp) !=0:
            tmp = tmp.set_index('AU_name')['Yield'].isna().groupby(level=0).sum()
            print(tmp)
            listRegionToDrop = tmp.index.to_list()
            stats = stats.drop(stats[(stats['Crop_ID'] == c) & (stats['AU_name'].isin(listRegionToDrop))].index)
        if verbose:
            print('**************')
            print(statsC['Crop_name'].iloc[0])
    #remove unamed columns, if any
    stats.drop(stats.filter(regex="Unnamed"), axis=1, inplace=True)
    if verbose:
        print('**************')
    if save_csv:
        stats.to_csv(os.path.join(config.output_dir, config.AOI + '_stats_missing_excluded.csv'), index=False)
    if plot_fig:
        #plot remaining regioms
        g = sns.relplot(
                data=stats,
                x="Year", y="Yield", col="AU_name",  hue="Crop_name",
                kind="line", linewidth=2, #zorder=5,
                col_wrap=5, height=1, aspect=2, #legend=False,
            )
        g.tight_layout()
        plt.savefig(os.path.join(config.output_dir, config.AOI + '_time_series_missing_excluded.png'))
    # print('end of b100 LoadLabel_Exclude_Missing')
    return stats

def LoadLabel(config, save_csv = True, plot_fig= False):
    '''
    This function is loading stats (without excluding admin with missing values)
    '''

    # get yield stats
    stats = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_STATS.csv'))
    #'AU_name' may not be present
    if ('AU_name' in stats.columns) == False:
        names = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_REGION_id.csv'))
        stats = pd.merge(stats, names, how='left', left_on=['Region_ID'], right_on=['AU_code'])
    if ('Crop_name' in stats.columns) == False:
        names = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_CROP_id.csv'))
        stats = pd.merge(stats, names, how='left', left_on=['Crop_ID'], right_on=['Crop_ID'])
    # drop years before period of interest
    stats = stats.drop(stats[stats['Year'] < config.year_start].index).sort_values('AU_name')
    stats.drop(stats.filter(regex="Unnamed"), axis=1, inplace=True)

    if save_csv:
        stats.to_csv(os.path.join(config.models_dir, config.AOI + '_stats.csv'), index=False)
    if plot_fig:
        #plot remaining regioms
        g = sns.relplot(
                data=stats,
                x="Year", y="Yield", col="AU_name",  hue="Crop_name",
                kind="line", linewidth=2, #zorder=5,
                col_wrap=5, height=1, aspect=2, #legend=False,
            )
        g.tight_layout()
        plt.savefig(os.path.join(config.models_dir, config.AOI + '_time_series.png'))
    # print('end of b100 LoadLabel_Exclude_Missing')
    return stats
