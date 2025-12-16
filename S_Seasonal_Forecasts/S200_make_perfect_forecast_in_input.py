import pandas as pd
import os
from S_Seasonal_Forecasts import date
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

'''
take extracted data,
add monthly observed cum rain and mean temp (computed from dekadal values) as forecasts,
save it as fn extraction + _ObsAsForecast
'''
# ####################### USER PARAM
# Filip's file
fn_extraction = r'V:\foodsec\Projects\GYF\ASAP extractions\GAUL0-MultipleCrop-ASAP-ASIS_v9.0.csv\GAUL0-MultipleCrop-ASAP-ASIS_v9.0.csv'
dateformat = '%Y%m%d' # GYF
group_variables = ['asap_unit_id', 'fao_unit_id', 'unit_name', 'extraction_setup_id', 'extraction_setup_name', 'indicator_id', 'indicator_name',
                   'mask_id', 'mask_name', 'mask_path']
name_date_column = 'dekad'
name_variable_column = 'indicator_name'
columns_to_keep_as999 = ["id", "adm_unit_layer", 'std',	'px_cnt_total', 'px_cnt_valid_data', 'px_cnt_valid_data_after_masking', 'px_cnt_weight_sum', 'px_cnt_weight_sum_used']

name_rainfall_var = 'ASAP:RAIN'
name_temperature_var = 'ASAP:TEMP'



# TUNISIA
# fn_extraction = r'V:\foodsec\Projects\SNYF\SIDv\TN\SF\NO_SF_baseline\Tuning_data\Multiple_WC-Tunisia-ASAP.csv'
# dateformat = '%Y-%m-%d' # SNYF
# group_variables = ['adm_name', 'adm_id', 'variable_name', 'var_id', 'classset_name','classesset_id', 'class_name', 'class_id']
# name_date_column = 'date'
# name_variable_column = 'variable_name'
# columns_to_keep_as999 = ["stddev", "afi_pct_used", "extraction_results_id"]
#
# name_rainfall_var = 'rainfall'
# name_temperature_var = 'temperature'

# ####################### END OF USER PARAM



group_variables.extend(['year', 'month'])
fn_out = os.path.splitext(fn_extraction)[0] + '_ObsAsForecast.csv' # debug
# #debug
# df = pd.read_csv(fn_out)
# df = df[df['unit_name']=='Algeria']
# fn_out = os.path.splitext(fn_extraction)[0] + '_ObsAsForecast_Algeria.csv' # debug
# df.to_csv(fn_out, index=False)
# #end of debug

df = pd.read_csv(fn_extraction)

# add days in dek as new column

df['date_dt'] = pd.to_datetime(df[name_date_column], format=dateformat)
df['DaysInDek'] = df['date_dt'].apply(date.get_dekade_days)
df['year'] = df['date_dt'].dt.year
df['month'] = df['date_dt'].dt.month


# compute monthly values
# Rain
rain = df[df[name_variable_column] == name_rainfall_var].copy()
monthly_r = rain.groupby(group_variables).agg(
        monthly_sum=('mean', 'sum'),                # ← the value we need
        first_date=('date_dt', 'min')               # ← first observed date
    )
monthly_r = monthly_r.rename_axis(index=group_variables).reset_index()
# fix names
monthly_r = monthly_r.rename(columns={"monthly_sum": "mean", "first_date": "date_dt"})
monthly_r[name_variable_column] = 'rainfall_monthly_sum'
# Temperature
temp = df[df[name_variable_column] == name_temperature_var].copy()
def agg_weighted_mean_and_first_date(sub):
    weighted_mean = (sub['mean']*sub['DaysInDek']).sum() / sub['DaysInDek'].sum()
    first_date = sub["date_dt"].min()
    return pd.Series({
        "weighted_mean": weighted_mean,
        "first_date": first_date
    })

monthly_t = temp.groupby(group_variables)\
                        .apply(agg_weighted_mean_and_first_date, include_groups=False)
monthly_t = monthly_t.rename_axis(index=group_variables).reset_index()
# fix names
monthly_t = monthly_t.rename(columns={"weighted_mean": "mean", "first_date": "date_dt"})
monthly_t[name_variable_column] = 'temperature_monthly_mean'
# concat it
monthly_rt = pd.concat([monthly_r, monthly_t], ignore_index=True)
# Keep only rows where the day of the month is 1 (drop incomplete records)
monthly_rt = monthly_rt[monthly_rt["date_dt"].dt.day == 1]

# add missing columns names
df.drop(["date_dt", "DaysInDek"], axis=1, inplace=True)
df.drop(['year', "month"], axis=1, inplace=True)

monthly_rt.drop(['year', "month"], axis=1, inplace=True)
monthly_rt[columns_to_keep_as999] = -999

df_forecast = pd.DataFrame(columns=df.columns) #empty df

# add fake forecast to orginal data
for v, var in zip(["r", "t"], ["rainfall_monthly_sum", "temperature_monthly_mean"]):
    tmp = monthly_rt[monthly_rt[name_variable_column] == var].copy()
    tmp[name_variable_column] = f'SF_{v}_1'
    df_forecast = pd.concat([df_forecast, tmp], ignore_index=True)
    for i in [1, 2, 3, 4, 5, 6]:
        tmp = monthly_rt[monthly_rt[name_variable_column] == var].copy()
        tmp[name_variable_column] = f'SF_{v}_{int(i) + 1}'
        tmp['date_dt'] = tmp['date_dt'] + pd.DateOffset(months=-int(i))
        df_forecast = pd.concat([df_forecast, tmp], ignore_index=True)

# add back to df
df_forecast[name_date_column] = df_forecast['date_dt'].dt.strftime(dateformat)
df_forecast.drop("date_dt", axis=1, inplace=True)
df = pd.concat([df, df_forecast], ignore_index=True)

df.to_csv(fn_out, index=False)



print('Finished')
#
#
# # debug, compare difference woth previous version
# csv1 = r'V:\foodsec\Projects\SNYF\SIDv\TN\SF\ObsAsSF\Tuning_data\Multiple_WC-Tunisia-ASAP_ObsAsForecast.csv'
# csv2 = r'V:\foodsec\Projects\SNYF\SIDv\TN\SF\NO_SF_baseline\Tuning_data\Multiple_WC-Tunisia-ASAP_ObsAsForecast2test.csv'
#
# df1 = pd.read_csv(csv1)
# df2 = pd.read_csv(csv2)
#
# print("Shape df1:", df1.shape)
# print("Shape df2:", df2.shape)
#
# print("\nColumns only in df1:", set(df1.columns) - set(df2.columns))
# print("Columns only in df2:", set(df2.columns) - set(df1.columns))