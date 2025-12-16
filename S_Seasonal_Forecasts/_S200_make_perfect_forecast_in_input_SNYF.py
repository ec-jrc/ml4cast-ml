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
# regNames = pd.read_csv(r'V:\foodsec\Projects\SNYF\SIDv\TN\SF\NO_SF_baseline\Tuning_data\TNWinter_REGION_id.csv')
# ####################### END OF USER PARAM

fn_out = os.path.splitext(fn_extraction)[0] + '_ObsAsForecast.csv'
df = pd.read_csv(fn_extraction)

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

df.to_csv(fn_out, index=False)



print('Finished')
