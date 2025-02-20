import pandas as pd
import numpy as np
import os

'''
Francesco extracts data for Gu and Deyr from Fews data (directly from Fews data Wearhouse, V:\foodsec\Projects\SNYF\stable_input_data\SO\FSNAU from Weston\FewsDW\SO_FDW_Jan28_2025.xlsx
and place it here
V:\foodsec\Projects\SNYF\stable_input_data\SO\Gu\Tuning_data\SOGut_FEWS_..
and same for Dyer
'''

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
# focus on Maize and Sorghum
crops = ['Maize', 'Sorghum', 'Cowpea']
seasons = ['Gu', 'Deyr']

for season in seasons:
    print(season)
    # READ teh file made by Francesco from FDW file
    if season == 'Gu':
        csv_in = r'V:\foodsec\Projects\SNYF\stable_input_data\SO\Gu\Tuning_data\SOGu_FEWS_STATS'
        fn_cropID2make = r'V:\foodsec\Projects\SNYF\stable_input_data\SO\Gu\Tuning_data\SOGu_CROP_id.csv'
    elif season == 'Deyr':
        csv_in = r'V:\foodsec\Projects\SNYF\stable_input_data\SO\Deyr\Tuning_data\SODeyr_FEWS_STATS'
        fn_cropID2make = r'V:\foodsec\Projects\SNYF\stable_input_data\SO\Deyr\Tuning_data\SODeyr_CROP_id.csv'
    else:
        exit()

    df = pd.read_csv(csv_in + '.csv')
    # remove space from crop name
    df.loc[df['Crop_name'] == 'Maize (Corn)', 'Crop_name'] = 'Maize'
    df.loc[df['Crop_name'] == 'Cowpea (unspecified)', 'Crop_name'] = 'Cowpea'
    # df[df['Crop_name'] == 'Maize (Corn)']['Crop_name'] = 'Maize'
    df = df[df['Crop_name'].isin(crops)]
    # drop everyhting that does not have area planted
    df = df[df['Area_type'] == 'P']
    # drop evrything that dos not have yield
    df = df.dropna(subset=['Yield'])
    df['Assigned_Crop_production_system'] = ''
    df_cleaned = df.drop(df.index)

    # for each admin I have to check what systems it has. Then I will two files one for agro and one for river
    # only all ??? check why (there should be none)
    # all + one (agro o river): consider that is all agro or river
    # all + the two: drop all (we don't know what represent) and keep agro or riverine
    for adm in df['adm_name'].unique():
        df_adm = df[df['adm_name'] == adm]
        for crop in df_adm['Crop_name'].unique():
            df_adm_crop = df_adm[df_adm['Crop_name'] == crop]
            df_adm_crop = df_adm_crop.sort_values("harvest_year", axis=0, ascending=True, na_position='last')
            systems = df_adm_crop['Crop_production_system'].unique()
            if len(systems) == 1:
                print('only one system, check, it will not be added to output')
                print(systems)
                print(df_adm_crop)
                print()
            elif len(systems) == 2:
                # print(systems)
                # I assume there is only one, turn All into that and remove duplicates
                named_system = [x for x in systems if x != 'All (PS)']
                # when both are present drop the 'All (PS)', I have seen cases where the all was wrong (see 2012 Baardheere, Maize)
                # Find all harvest_years where both "All" and "river" exist
                years_with_both_systems = df_adm_crop[df_adm_crop['Crop_production_system'] == named_system[0]]['harvest_year'].unique()
                df_adm_crop = df_adm_crop[
                    ~((df_adm_crop['harvest_year'].isin(years_with_both_systems)) & (
                            df_adm_crop['Crop_production_system'] == 'All (PS)'))]
                df_adm_crop['Assigned_Crop_production_system'] = named_system[0]

                df_cleaned = pd.concat([df_cleaned, df_adm_crop])
                # print()
            elif len(systems) == 3:
                # print(systems)
                # print(df_adm_crop)
                df_adm_crop = df_adm_crop[df_adm_crop['Crop_production_system'] != 'All (PS)']
                df_adm_crop['Assigned_Crop_production_system'] = df_adm_crop['Crop_production_system']
                df_cleaned = pd.concat([df_cleaned, df_adm_crop])
                # print()
            else:
                print('n sistem is not 1, 2 or 3, check')
                print(systems)
                print(df_adm_crop)
                print()
    # here I have only data with planted area and the two named systems
    # add the name of system to crop and save
    df_cleaned['Crop_name'] = df_cleaned['Crop_name'] + '_' + df_cleaned['Assigned_Crop_production_system']
    # add crop IDS for new crops (and save SOGu_CROP_id.csv)
    elements = np.sort(df_cleaned.Crop_name.unique()).tolist()
    element_dict = {element: i + 1 for i, element in enumerate(elements)}
    df_cleaned['Crop_ID'] = df_cleaned['Crop_name'].map(element_dict)
    df_cleaned = df_cleaned.rename(columns={'harvest_year': 'Year'})
    df_cleaned.to_csv(csv_in + '_only_named_systems.csv', index=False)
    if os.path.exists(fn_cropID2make):
        directory = os.path.dirname(fn_cropID2make)
        file_name = os.path.basename(fn_cropID2make)
        if not os.path.exists(os.path.join(directory, '_' + file_name)):
            os.rename(fn_cropID2make, os.path.join(directory, '_' + file_name))
    # make it and save it
    dfid = pd.DataFrame(list(element_dict.items()), columns=['Crop_name', 'Crop_ID'])
    dfid = dfid[dfid.columns[::-1]]
    dfid.to_csv(fn_cropID2make, index=False)


print('End')