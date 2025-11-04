import pandas as pd
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

# fn1 = r'V:\foodsec\Projects\SNYF\SIDv\TN\_SF_test\ObsAsSF\Tuning_data\TNWinter_STATS_cleaned90.csv'
# fn2 = r'V:\foodsec\Projects\SNYF\SIDv\TN\_SF_test\NO_SF_baseline\Tuning_data\TNWinter_STATS_cleaned90.csv'

fn1 = r'V:\foodsec\Projects\SNYF\SIDv\TN\SF\ObsAsSF\Tuning_data\Multiple_WC-Tunisia-ASAP_ObsAsForecast.csv'
fn2 = r'V:\foodsec\Projects\SNYF\SIDv\TN\SF\NO_SF_baseline\Tuning_data\Multiple_WC-Tunisia-ASAP.csv'


df1 = pd.read_csv(fn1)
df2 = pd.read_csv(fn2)

# adjust for asap data comparison
if True:
    df1 = df1.drop(['adm_name','fnid'], axis=1)
    df2 = df2.drop(['adm_name'], axis=1)
    print(df1.info())
    print(df2.info())
    df1['extraction_results_id'] = df1['extraction_results_id'].fillna(0)
    df1['var_id'] = df1['var_id'].fillna(0)
    df1['class_id'] = df1['class_id'].fillna(0)
    df1['classesset_id'] = df1['classesset_id'].fillna(0)
    df1['stddev'] = df1['stddev'].fillna(0)
    df1['afi_pct_used'] = df1['afi_pct_used'].fillna(0)

    df1['date'] = pd.to_datetime(df1['date'])
    df2['date'] = pd.to_datetime(df2['date'])

    for col in df1.columns:
        print(col)
        df1[col] = df1[col].astype(df2[col].dtype)

    # Round all float columns to 6 decimal places
    float_cols_df1 = df1.select_dtypes(include=['float64', 'float32']).columns
    float_cols_df2 = df2.select_dtypes(include=['float64', 'float32']).columns

    df1[float_cols_df1] = df1[float_cols_df1].round(6)
    df2[float_cols_df2] = df2[float_cols_df2].round(6)


df1_sorted = df1.sort_values(by=df1.columns.tolist()).reset_index(drop=True)
df2_sorted = df2.sort_values(by=df2.columns.tolist()).reset_index(drop=True)

df_diff = pd.merge(df1_sorted, df2_sorted, indicator=True, how='outer')
df_diff = df_diff[df_diff['_merge'] != 'both']
print('Differences')
print(df_diff.head())

# Rows only in df1
only_in_df1 = df_diff[df_diff['_merge'] == 'left_only'].drop('_merge', axis=1)
print(f"Rows only in df1: {len(only_in_df1)}")
print(only_in_df1.head())

print('********* only in df1 removing SF')
print(only_in_df1[~only_in_df1['variable_name'].str.startswith('SF')])

# Rows only in df2
only_in_df2 = df_diff[df_diff['_merge'] == 'right_only'].drop('_merge', axis=1)
print(f"\nRows only in df2: {len(only_in_df2)}")
print(only_in_df2.head())


