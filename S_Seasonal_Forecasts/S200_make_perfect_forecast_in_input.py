import pandas as pd
import os
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

'''
take extracted data, 
add observed rain and prec (taken from monthly features) as forecasts,
save it as fn extraction + _ObsAsForecast
'''
# ZA
# fn_extraction = r'V:\foodsec\Projects\SNYF\SIDv\ZA\SF\Tuning_data\Maize_(corn)_WC-South_Africa-ASAP.csv'
# regNames = pd.read_csv(r'V:\foodsec\Projects\SNYF\SIDv\ZA\SF\Tuning_data\ZAsummer_REGION_id.csv')
# df_mon = pd.read_csv(r'V:\foodsec\Projects\SNYF\SIDv\ZA\SF\RUN_Maize_(corn)_WC-South_Africa-ASAP\TUNE_ZAvSeas5base\ZAsummer_monthly_features.csv')
# TUNISIA
fn_extraction = r'V:\foodsec\Projects\SNYF\SIDv\TN\Winter2\Tuning_data\Multiple_WC-Tunisia-ASAP.csv'
regNames = pd.read_csv(r'V:\foodsec\Projects\SNYF\SIDv\TN\Winter2\Tuning_data\TNWinter_REGION_id.csv')
df_mon = pd.read_csv(r'V:\foodsec\Projects\SNYF\SIDv\TN\Winter2\RUN_Multiple_WC-Tunisia-ASAP\TUNE_TNv_20250704\TNWinter_monthly_features.csv')


fn_out = os.path.splitext(fn_extraction)[0] + '_ObsAsForecast.csv'

df_dek = pd.read_csv(fn_extraction)
# do as in standard code, replace names with region_id

# now link it with adm_id and name
df_dek = pd.merge(df_dek, regNames, left_on=['adm_id'], right_on=['adm_id'])
# remove the admin name form extraction, not needed, use the one from region file
df_dek = df_dek.rename(columns={'adm_name_x': 'adm_name_extraction', 'adm_name_y': 'adm_name'})
df_dek.drop('adm_name_extraction', axis=1, inplace=True)
# get value of columns I will need to fill
classset_name = df_dek.classset_name.iloc[0]
class_name = df_dek.class_name.iloc[0]

# get monthly features
# A given month MM contains means of that month MM with date MM/01,
# when preparing the data (load), this has to be assigned to the previous month

df_mon = df_mon[(df_mon.variable_name == 'rainfall') | (df_mon.variable_name == 'temperature')]
df_mon = df_mon.sort_values(by=['variable_name', 'adm_name', 'Date'])
res = []
detection = 0
for name, group in df_mon.groupby(['variable_name', 'adm_name', 'Date']):
    # Get the current date and the following 6 dates
    current_date = group['Date'].iloc[0]
    next_dates = df_mon[(df_mon['adm_name'] == name[1]) & (df_mon['Date'] > current_date) & (df_mon['variable_name'] == name[0])].sort_values(by='Date').head(6)
    combined_dates = pd.concat([group, next_dates])
    # # shift - 1 month REMOVED, THIS HAS TO BE DONE IN LOAD
    # combined_dates['Date_datetime'] = pd.to_datetime(combined_dates['Date'])
    # combined_dates['Date'] = (combined_dates['Date_datetime'] + pd.DateOffset(months=-1)).dt.strftime('%Y-%m-%d')
    desired_columns = combined_dates[['variable_name', 'adm_name', 'adm_id', 'Date', 'mean']]
    desired_columns = desired_columns.rename(columns={'variable_name': 'variable_name_original'})

    if len(desired_columns) != 7:
        print('Warning: there is no 7 months ahead for date ' + desired_columns.Date.iloc[0])
        detection = 1

    # Add a column with an increasing number
    desired_columns['variable_name'] = range(1, len(desired_columns) + 1)
    # use first character, r or t
    desired_columns['variable_name'] = 'SF_' + name[0][0] + '_' + desired_columns['variable_name'].astype(str)
    desired_columns['Date'] = desired_columns['Date'].iloc[0]
    res.append(desired_columns)
if detection != 0:
    print ('The warning is not a problem if it is the last year')
result_df = pd.concat(res, ignore_index=True)
result_df = result_df.drop('variable_name_original', axis=1)
result_df = result_df.rename(columns={'Date': 'date'})
df_dek = pd.concat([df_dek, result_df], axis=0)
df_dek = df_dek.sort_values(by=[ 'adm_name', 'date'])
df_dek.classset_name = classset_name
df_dek.class_name = class_name

df_dek.to_csv(fn_out, index=False)

print('Finished')
