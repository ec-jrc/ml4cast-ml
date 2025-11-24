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
# TUNISIA
fn_extraction = r'V:\foodsec\Projects\SNYF\SIDv\TN\SF\NO_SF_baseline\Tuning_data\Multiple_WC-Tunisia-ASAP.csv'
regNames = pd.read_csv(r'V:\foodsec\Projects\SNYF\SIDv\TN\SF\NO_SF_baseline\Tuning_data\TNWinter_REGION_id.csv')
# ####################### END OF USER PARAM

fn_out = os.path.splitext(fn_extraction)[0] + '_ObsAsForecast.csv'
df = pd.read_csv(fn_extraction)
# # do as in standard code, replace names with region_id
# # and link it with adm_id and name
# df = pd.merge(df, regNames, left_on=['adm_id'], right_on=['adm_id'])
# # remove the admin name from extraction, not needed, use the one from region file
# df = df.rename(columns={'adm_name_x': 'adm_name_extraction', 'adm_name_y': 'adm_name'})
# df.drop('adm_name_extraction', axis=1, inplace=True)

# get value of columns I will need to fill
classset_name = df.classset_name.iloc[0]
class_name = df.class_name.iloc[0]

# add days in dek as new column
df['date_dt'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['DaysInDek'] = df['date_dt'].apply(date.get_dekade_days)

# compute monthly values
# Rain
rain = df[df['variable_name'] == 'rainfall'].copy()
monthly_r = rain.groupby([rain['adm_name'], rain['adm_id'], rain['variable_name'], rain['var_id'], rain['classset_name'],
                        rain['classesset_id'], rain['class_name'], rain['class_id'],
                        rain['date_dt'].dt.year, rain['date_dt'].dt.month]).agg(
        monthly_sum=('mean', 'sum'),                # ← the value we need
        first_date=('date_dt', 'min')               # ← first observed date
    )
monthly_r = monthly_r.rename_axis(index=["adm_name", "adm_id", "variable_name", "var_id", "classset_name", "classesset_id", "class_name", "class_id", "year", "month"]).reset_index()
# fix names
monthly_r = monthly_r.rename(columns={"monthly_sum": "mean", "first_date": "date_dt"})
monthly_r['variable_name'] = 'rainfall_monthly_sum'
# Temperature
temp = df[df['variable_name'] == 'temperature'].copy()
def agg_weighted_mean_and_first_date(sub):
    weighted_mean = (sub['mean']*sub['DaysInDek']).sum() / sub['DaysInDek'].sum()
    first_date = sub["date_dt"].min()
    return pd.Series({
        "weighted_mean": weighted_mean,
        "first_date": first_date
    })

monthly_t = temp.groupby([temp['adm_name'], temp['adm_id'], temp['variable_name'], temp['var_id'], temp['classset_name'],
                        temp['classesset_id'], temp['class_name'], temp['class_id'], temp['date_dt'].dt.year, temp['date_dt'].dt.month])\
                        .apply(agg_weighted_mean_and_first_date, include_groups=False)
monthly_t = monthly_t.rename_axis(index=["adm_name", "adm_id", "variable_name", "var_id", "classset_name", "classesset_id", "class_name", "class_id", "year", "month"]).reset_index()
# fix names
monthly_t = monthly_t.rename(columns={"weighted_mean": "mean", "first_date": "date_dt"})
monthly_t['variable_name'] = 'temperature_monthly_mean'
# concat it
monthly_rt = pd.concat([monthly_r, monthly_t], ignore_index=True)
# Keep only rows where the day of the month is 1 (drop incomplete records)
monthly_rt = monthly_rt[monthly_rt["date_dt"].dt.day == 1]

# add missing columns names
df.drop(["date_dt", "DaysInDek"], axis=1, inplace=True)
monthly_rt.drop(['year', "month"], axis=1, inplace=True)
monthly_rt["stddev"] = -999
monthly_rt["afi_pct_used"] = -999
monthly_rt["extraction_results_id"] = -999

df_forecast = pd.DataFrame(columns=df.columns) #empty df
# add fake forecast


# pandas version
for v, var in zip(["r", "t"], ["rainfall_monthly_sum", "temperature_monthly_mean"]):
    tmp = monthly_rt[monthly_rt['variable_name'] == var].copy()
    tmp['variable_name'] = f'SF_{v}_1'
    df_forecast = pd.concat([df_forecast, tmp], ignore_index=True)
    for i in [1, 2, 3, 4, 5, 6]:
        tmp = monthly_rt[monthly_rt['variable_name'] == var].copy()
        tmp['variable_name'] = f'SF_{v}_{int(i) + 1}'
        tmp['date_dt'] = tmp['date_dt'] + pd.DateOffset(months=-int(i))
        df_forecast = pd.concat([df_forecast, tmp], ignore_index=True)

# add back to df
df_forecast['date'] = df_forecast['date_dt'].dt.strftime("%Y-%m-%d")
df_forecast.drop("date_dt", axis=1, inplace=True)
df = pd.concat([df, df_forecast], ignore_index=True)
# df_forecast.drop('Date_dt', axis=1, inplace=True)
# df_forecast.drop('DaysInDek', axis=1, inplace=True)
df.to_csv(fn_out, index=False)


# for v, var in zip(["r", "t"], ["rainfall_monthly_sum", "temperature_monthly_mean"]):
#     for adm in monthly_rt["adm_name"].unique():
#         for date in monthly_rt['date_dt'].unique():
#             tmp = monthly_rt[(monthly_rt['variable_name'] == var) & (monthly_rt['date_dt'] == date) & (monthly_rt['adm_name'] == adm)].copy()
#             if len(tmp) > 1:
#                 print('problem')
#             for i in [0, 1, 2, 3, 4, 5, 6]:
#                 tmp['variable_name'] = f'SF_{v}{int(i)+1}'
#                 # here I have the var (say r) for a given month M (say 6),
#                 # this has to be saved as horizon 1 for this Month, 2 for M-1, 3 for M-2m ..
#                 tmp['date_dt'] = tmp['date_dt'] + pd.DateOffset(months=-int(i))
#                 df_forecast = pd.concat([df_forecast, tmp], ignore_index=True)

# compute  monthly features
# A given month MM contains means of that month MM with date MM/01,
# when preparing the data (load), this has to be assigned to the previous month


# print('Debug rewrite')
#
# df_mon = df_mon[(df_mon.variable_name == 'rainfall') | (df_mon.variable_name == 'temperature')]
# df_mon = df_mon.sort_values(by=['variable_name', 'adm_name', 'Date'])
# res = []
# detection = 0
# for name, group in df_mon.groupby(['variable_name', 'adm_name', 'Date']):
#     # Get the current date and the following 6 dates
#     current_date = group['Date'].iloc[0]
#     next_dates = df_mon[(df_mon['adm_name'] == name[1]) & (df_mon['Date'] > current_date) & (df_mon['variable_name'] == name[0])].sort_values(by='Date').head(6)
#     combined_dates = pd.concat([group, next_dates])
#     # # shift - 1 month REMOVED, THIS HAS TO BE DONE IN LOAD
#     # combined_dates['Date_datetime'] = pd.to_datetime(combined_dates['Date'])
#     # combined_dates['Date'] = (combined_dates['Date_datetime'] + pd.DateOffset(months=-1)).dt.strftime('%Y-%m-%d')
#     desired_columns = combined_dates[['variable_name', 'adm_name', 'adm_id', 'Date', 'mean']]
#     desired_columns = desired_columns.rename(columns={'variable_name': 'variable_name_original'})
#
#     if len(desired_columns) != 7:
#         print('Warning: there is no 7 months ahead for date ' + desired_columns.Date.iloc[0])
#         detection = 1
#
#     # Add a column with an increasing number
#     desired_columns['variable_name'] = range(1, len(desired_columns) + 1)
#     # use first character, r or t
#     desired_columns['variable_name'] = 'SF_' + name[0][0] + '_' + desired_columns['variable_name'].astype(str)
#     desired_columns['Date'] = desired_columns['Date'].iloc[0]
#     res.append(desired_columns)
# if detection != 0:
#     print ('The warning is not a problem if it is the last year')
# result_df = pd.concat(res, ignore_index=True)
# result_df = result_df.drop('variable_name_original', axis=1)
# result_df = result_df.rename(columns={'Date': 'date'})
# df_dek = pd.concat([df_dek, result_df], axis=0)
# df_dek = df_dek.sort_values(by=[ 'adm_name', 'date'])
# df_dek.classset_name = classset_name
# df_dek.class_name = class_name
#
# df_dek.to_csv(fn_out, index=False)

print('Finished')
