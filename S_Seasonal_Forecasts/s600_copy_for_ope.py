import pandas as pd

pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', None)

# fill new 2026 data of ObsAsForecast and SfAsForecast with previous year to allow OPE

fn_in = r'V:\foodsec\Projects\SNYF\SIDvs\ZA\summer2025data\SF2\Tuning_data\Maize_(corn)_WC-South_Africa-ASAP_SfAsForecast.csv'
fn_out = r'V:\foodsec\Projects\SNYF\SIDvs\ZA\summer2025data\SF2\OpeForecast_data\Maize_(corn)_WC-South_Africa-ASAP_SfAsForecast.csv'

df = pd.read_csv(fn_in)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df_original = df.copy()

for variable_name in df.variable_name.unique():
    print(variable_name)
    for adm_id in df.adm_id.unique():
        df_var_ad = df[(df['variable_name'] == variable_name) & (df['adm_id'] == adm_id)]
        # get last date
        most_recent = df_var_ad["date"].max()
        new_date = pd.Timestamp(year=2026, month=2, day=1)
        print(most_recent, new_date)
        most_recent_1y = most_recent - pd.DateOffset(years=1)
        new_date_1y = new_date - pd.DateOffset(years=1)
        print(most_recent_1y, new_date_1y)

        mask = (df_var_ad["date"] > most_recent_1y) & (df_var_ad["date"] <= new_date_1y)
        df_selected = df_var_ad.loc[mask]
        df_selected = df_selected.copy()
        df_selected["date"] = df_selected["date"] + pd.DateOffset(years=1)
        df_original = pd.concat([df_original, df_selected], ignore_index=True)


df_original.to_csv(fn_out)
print('End')