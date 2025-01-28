import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

csv_in = r'V:\foodsec\Projects\SNYF\stable_input_data\SO\Gu\Tuning_data\SOGu_STATS'
# csv_in = r'V:\foodsec\Projects\SNYF\stable_input_data\SO\Deyr\Tuning_data\SODeyr_STATS'
df = pd.read_csv(csv_in + '.csv')

# focus on Maize
df = df[df['Crop_name'] == 'Maize']
df = df.replace({'none': None})
# keep data after 2000 (included)
startYear = 1900

# First we make a dataset with all the missing years present as empty rows (note that when there ais
# one occurrence of thw different cropping system we requiore them to be present evry year after the
# 'none' series


adm_ids = df['adm_id'].unique()
print('number of admin units = ' + str(len(adm_ids)))
df_out = df.iloc[:0]
df_out['Crop_production_system_2'] = np.NaN

for adm_id in adm_ids:
    df_adm_id = df[df['adm_id'] == adm_id]
    years = np.arange(df_adm_id['Year'].min(), df_adm_id['Year'].max() + 1)
    systems = df_adm_id['Crop_production_system'].unique()
    other_sys = [x for x in systems if x != 'none']
    # I want one record per year at minimum in case the admin has only none or none and one system. If the admin has two systems (and can
    # have none too) I want one record per year at min (if the year was fully missing), but two if the year had one or the other sysyem
    # Add rows for missing years
    # Create a complete DataFrame of all combinations of adm_id and year
    complete_index = pd.MultiIndex.from_product([years], names=['Year'])
    complete_df_adm_id = pd.DataFrame(index=complete_index).reset_index()
    # Merge the complete DataFrame with the original DataFrame
    merged_df_adm_id = pd.merge(complete_df_adm_id, df_adm_id, on=['Year'], how='left')
    merged_df_adm_id['Crop_production_system_2'] = None #np.NaN #systems[0]
    merged_df_adm_id['Crop_production_system_2'].update(merged_df_adm_id['Crop_production_system'])
    # If the two systems are present once, make sure that when it is not None,I have two rows for the two systems (i.e.
    # add the missing if missing
    if len(systems) == 3:
        # Get unique years
        unique_years = merged_df_adm_id['Year'].unique()
        # Define the required crop production systems
        required_systems = [element for element in systems if element is not None] #['agro_pastoral', 'riverine']
        # Prepare a list to collect rows
        rows_to_add = []
        # Identify the columns to fill with NaN
        columns_to_fill = [col for col in merged_df_adm_id.columns if col not in ['Year', 'Crop_production_system_2']]
        # Iterate over each unique year
        for year in years:
            # Filter rows for the current year
            year_df = merged_df_adm_id[merged_df_adm_id['Year'] == year]
            # Check that if there is only one and it is None, do nothing
            if not(len(year_df) == 1 and year_df['Crop_production_system_2'].unique()[0] == None):
                # Check for 'agro_pastoral'
                if 'agro_pastoral' not in year_df['Crop_production_system_2'].values:
                    # Add a missing row
                    row = {'Year': year, 'Crop_production_system_2': 'agro_pastoral'}
                    row.update({col: np.nan for col in columns_to_fill})
                    rows_to_add.append(row)
                # Check for 'riverine'
                if 'riverine' not in year_df['Crop_production_system_2'].values:
                    # Add a missing row
                    row = {'Year': year, 'Crop_production_system_2': 'riverine'}
                    row.update({col: np.nan for col in columns_to_fill})
                    rows_to_add.append(row)

        # Append the missing rows to the original DataFrame
        # Create a DataFrame from the rows to add
        additional_df = pd.DataFrame(rows_to_add)

        # Concatenate the original DataFrame with the new rows
        merged_df_adm_id = pd.concat([merged_df_adm_id, additional_df], ignore_index=True)

        # Sort by Year and Crop_production_system for clarity
        merged_df_adm_id = merged_df_adm_id.sort_values(by=['Year', 'Crop_production_system_2'])



    # Fill missing info
    # Columns to fill
    columns_to_fill = ['adm_id', 'Crop_ID', 'adm_name', 'Crop_name', 'fnid']
    # Forward and backward fill the missing values
    merged_df_adm_id[columns_to_fill] = merged_df_adm_id[columns_to_fill].ffill()
    merged_df_adm_id[columns_to_fill] = merged_df_adm_id[columns_to_fill].bfill()
    df_out = pd.concat([df_out, merged_df_adm_id])
    # # I have three possible cases
    # if len(systems) == 1: #only one system, should be only "none"
    #     # Add rows for missing years
    #     # Create a complete DataFrame of all combinations of adm_id and year
    #     complete_index = pd.MultiIndex.from_product([years], names=['Year'])
    #     complete_df_adm_id = pd.DataFrame(index=complete_index).reset_index()
    #     # Merge the complete DataFrame with the original DataFrame
    #     merged_df_adm_id = pd.merge(complete_df_adm_id, df_adm_id, on=['Year'], how='left')
    #     merged_df_adm_id['Crop_production_system_2'] = systems[0]
    #     # Fill missing info
    #     # Columns to fill
    #     columns_to_fill = ['adm_id', 'Crop_ID', 'adm_name', 'Crop_name', 'fnid']
    #     # Forward and backward fill the missing values
    #     merged_df_adm_id[columns_to_fill] = merged_df_adm_id[columns_to_fill].ffill()
    #     merged_df_adm_id[columns_to_fill] = merged_df_adm_id[columns_to_fill].bfill()
    #     df_out = pd.concat([df_out, merged_df_adm_id])
    # elif len(systems) == 2: # there is 'none' initially and then one of the two (agro_pastoral OR riverine)
    #     # Add rows for missing years
    #     # Create a complete DataFrame of all combinations of adm_id and year
    #     complete_index = pd.MultiIndex.from_product([years], names=['Year'])
    #     complete_df_adm_id = pd.DataFrame(index=complete_index).reset_index()
    #     # Merge the complete DataFrame with the original DataFrame
    #     merged_df_adm_id = pd.merge(complete_df_adm_id, df_adm_id, on=['Year'], how='left')
    #     merged_df_adm_id['Crop_production_system_2'] = other_sys[0]
    #     # Fill missing info
    #     # Columns to fill
    #     columns_to_fill = ['adm_id', 'Crop_ID', 'adm_name', 'Crop_name', 'fnid']
    #     # Forward and backward fill the missing values
    #     merged_df_adm_id[columns_to_fill] = merged_df_adm_id[columns_to_fill].ffill()
    #     merged_df_adm_id[columns_to_fill] = merged_df_adm_id[columns_to_fill].bfill()
    #     df_out = pd.concat([df_out, merged_df_adm_id])
    # elif len(systems) == 3: # there is 'none' initially and then both agro_pastoral OR riverine
    #     # add missing records to the 'none' part (make sure no gaps between years[0] and the year before the first not 'none'
    #     last_year_to_be_present = df_adm_id[df_adm_id['Crop_production_system'] != 'none']['Year'].min()-1
    #     #make sure it is equal or after the last none
    #     if last_year_to_be_present < df_adm_id[df_adm_id['Crop_production_system'] == 'none']['Year'].max():
    #         print('assumption that crop system is only after none violated here. check')
    #     df_adm_id_none = df_adm_id[df_adm_id['Crop_production_system'] == 'none']
    #     complete_index = pd.MultiIndex.from_product([np.arange(years[0], last_year_to_be_present+1)], names=['Year'])
    #     complete_df = pd.DataFrame(index=complete_index).reset_index()
    #     # Merge the complete DataFrame with the original DataFrame
    #     df_adm_id_none = pd.merge(complete_df, df_adm_id_none, on=['Year'], how='left')
    #     df_adm_id_none['Crop_production_system_2'] = 'none'
    #     # Fill missing info
    #     columns_to_fill = ['adm_id', 'Crop_ID', 'adm_name', 'Crop_name', 'fnid']
    #     df_adm_id_none[columns_to_fill] = df_adm_id_none[columns_to_fill].ffill()
    #     df_adm_id_none[columns_to_fill] = df_adm_id_none[columns_to_fill].bfill()
    #     df_out = pd.concat([df_out, df_adm_id_none])
    #
    #     # Now treat the mixed part
    #     for sys in other_sys:
    #         first_year_to_be_present = df_adm_id[df_adm_id['Crop_production_system'] == 'none']['Year'].max() + 1
    #         # make sure that there is no sys data before that year
    #         if first_year_to_be_present > df_adm_id[df_adm_id['Crop_production_system'] == sys]['Year'].min():
    #             print('assumption that crop system is only after none violated here (case two systems). check')
    #         df_adm_id_sys = df_adm_id[df_adm_id['Crop_production_system'] == sys]
    #         complete_index = pd.MultiIndex.from_product([np.arange(first_year_to_be_present, years[-1] + 1)], names=['Year'])
    #         complete_df = pd.DataFrame(index=complete_index).reset_index()
    #         # Merge the complete DataFrame with the original DataFrame
    #         df_adm_id_sys = pd.merge(complete_df, df_adm_id_sys, on=['Year'], how='left')
    #         df_adm_id_sys['Crop_production_system_2'] = sys
    #         # Fill missing info
    #         columns_to_fill = ['adm_id', 'Crop_ID', 'adm_name', 'Crop_name', 'fnid']
    #         df_adm_id_sys[columns_to_fill] = df_adm_id_sys[columns_to_fill].ffill()
    #         df_adm_id_sys[columns_to_fill] = df_adm_id_sys[columns_to_fill].bfill()
    #         df_out = pd.concat([df_out, df_adm_id_sys])
    # else:
    #     print('case not foreseen, check')
df_out = df_out[df_out['Year'] >= startYear]
df_out.sort_values(by = ['adm_id', 'Year', 'Crop_production_system_2'], ascending=[True, True, True], inplace=True)
df_out.to_csv(csv_in + '_with_missing_as_empty_rows.csv', index = False)
df_out_missing = df_out[df_out['Yield'].isnull()]

rang = range(df_out_missing['Year'].min(), df_out_missing['Year'].max()+2)
plt.hist(df_out_missing['Year'], bins=rang, align='left')
plt.xticks(rang)
plt.xticks(rotation=90)
#if you just want to look at the plot
# plt.show()
#if you want to save the plot to a file
plt.savefig(csv_in + '_missing_histogram.png')


