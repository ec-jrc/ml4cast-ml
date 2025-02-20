import pandas as pd
import numpy as np


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
# focus on Maize and Sorghum
crops = ['Maize', 'Sorghum']
seasons = ['Gu', 'Deyr']
for season in seasons:
    print(season)
    #READ HARVEST
    if season == 'Gu':
        csv_in = r'V:\foodsec\Projects\SNYF\stable_input_data\SO\Gu\Tuning_data\SOGu_STATS'
        year_dff_column_for_merge = 'Year_FSNAU'
        dropSeason = 'Deyr'
        areaH_col = 'Area Planted Gu (ha)'
    elif season == 'Deyr':
        csv_in = r'V:\foodsec\Projects\SNYF\stable_input_data\SO\Deyr\Tuning_data\SODeyr_STATS'
        year_dff_column_for_merge = 'Year_harv_Deyr'
        dropSeason = 'Gu'
        areaH_col = 'Area Planted_Deyr(ha)'
    else:
        exit()
    dfh = pd.read_csv(csv_in + '.csv', thousands=',')
    #READ FSNAU
    # our not updated file
    # dff = pd.read_excel(r'V:\foodsec\Projects\SNYF\stable_input_data\SO\Updated FSNAU yield Estimates v2.10.02.2017_no_merged_cells.xlsx')#, thousands=',')
    # updated file sent by Weston 2025 02 05 ( a fix versioon of it becaus ethe orginal had formulas and frezed cells)
    dff = pd.read_excel(
        r'V:\foodsec\Projects\SNYF\stable_input_data\SO\FSNAU from Weston\19140Deyr 2023-24_Draft 3_13012024_no_merged_cells_fix.xlsx')  # , thousands=',')
    #read name LUT
    df_lut = pd.read_excel(r'V:\foodsec\Projects\SNYF\stable_input_data\SO\FSNAU Hravest name corrisp.xlsx')
    dict_lut = dict(zip(df_lut['NamesFEWS'], df_lut['NamesFSNAU']))
    dfh['adm_name_FSNAU_style'] = dfh['adm_name']
    dfh['adm_name_FSNAU_style'] = dfh['adm_name_FSNAU_style'].replace(dict_lut)
    #  remove Districts Total and all
    dff = dff[dff['District'] != 'Total']
    dff = dff[dff['District'] != 'All']
    dff = dff.rename(columns={'Year': 'Year_FSNAU'})
    # focus on crops
    dfh = dfh[dfh['Crop_name'].isin(crops)]
    dff = dff[dff['CROP'].isin(crops)]
    # Keep only main (Gu and Deyr)
    dff = dff[dff.columns.drop(list(dff.filter(regex="Off")))]
    dff = dff[dff.columns.drop(list(dff.filter(regex="off")))]
    # Production is 0 instead of NaN
    # Set various production to NaN unless harvested area or yield are 0
    dff.loc[(dff['Prodn GU(MT)'] == 0) & ((dff['Area Harvstd Gu_(Ha)'].isnull()) | (dff['Yield GU'].isnull())), 'Prodn GU(MT)'] = np.NAN
    # dff.loc[(dff['Prodn GU Off(MT)'] == 0) & ((dff['Area Harvstd Gu Offsn_(Ha)'].isnull()) | (dff['Yield GU Off'].isnull())), 'Prodn GU Off(MT)'] = np.NAN
    # dff.loc[(dff['Total Area Planted (Gu+Offsn)'] == 0) & ((dff['Area Planted Gu (ha)'].isnull()) & (dff['Area Planted Gu Offsn (ha)'].isnull())), 'Total Area Planted (Gu+Offsn)'] = np.NAN
    # dff.loc[(dff['Total Area Harvested (Gu+Offsn)'] == 0) & ((dff['Area Harvstd Gu_(Ha)'].isnull()) & (dff['Area Harvstd Gu_(Ha)'].isnull())), 'Total Area Harvested (Gu+Offsn)'] = np.NAN
    # dff.loc[(dff['Total Prodn(MT)- Gu+Offsn'] == 0) & ((dff['Prodn GU(MT)'].isnull()) & (dff['Prodn GU Off(MT)'].isnull())), 'Total Prodn(MT)- Gu+Offsn'] = np.NAN

    dff.loc[(dff['Prodn_Deyr(MT)'] == 0) & ((dff['Area Harvstd_Deyr(ha)'].isnull()) | (dff['Yield_Deyr'].isnull())), 'Prodn_Deyr(MT)'] = np.NAN
    # dff.loc[(dff['Prodn_Deyr Off(MT)'] == 0) & ((dff['Area Harvstd_Deyr Offsn (ha)'].isnull()) | (dff['Yield_Deyr Off'].isnull())), 'Prodn_Deyr Off(MT)'] = np.NAN
    # dff.loc[(dff['Total Area Planted (Deyr+Offsn)'] == 0) & ((dff['Area Planted_Deyr(ha)'].isnull()) & (dff['Area Planted_Deyr Offsn (ha)'].isnull())), 'Total Area Planted (Deyr+Offsn)'] = np.NAN
    # dff.loc[(dff['Total Area Harvested (Deyr+Offsn)'] == 0) & ((dff['Area Harvstd_Deyr(ha)'].isnull()) & (dff['Area Harvstd_Deyr Offsn (ha)'].isnull())), 'Total Area Harvested (Deyr+Offsn)'] = np.NAN
    # dff.loc[(dff['Total Prodn(MT)- Deyr+Offsn'] == 0) & ((dff['Prodn_Deyr(MT)'].isnull()) & (dff['Prodn_Deyr Off(MT)'].isnull())), 'Total Prodn(MT)- Deyr+Offsn'] = np.NAN

    # drop unnecessary cols
    dff.drop(['Area Planted (ha)', 'Area Harvested (ha)', 'Production in MT'], axis=1, inplace=True) #, 'Yield - Gu + offsn', 'Yield - Deyr + offsn']

    # make it compatible with FSNAU
    dfh = dfh.replace({'none': 'Combined'})
    dfh = dfh.replace({'riverine': 'Riverine'})
    dfh = dfh.replace({'agro_pastoral': 'Agropastoral'})

    dff['Year_harv_Deyr'] = dff['Year_FSNAU'] + 1
    # limit harvest o 2016 because last year of FSNAU
    #dfh = dfh[dfh['Year'] <= 2016]

    df = dff.merge(dfh, left_on=[year_dff_column_for_merge, 'District', 'CROP', 'Livelihood System'], right_on=['Year', 'adm_name_FSNAU_style', 'Crop_name', 'Crop_production_system'], how='outer')
    df = df[df.columns.drop(list(df.filter(regex=dropSeason)))]
    # drop rows where all stat values are empty (there is only region and year in fsnau)
    cols = df.columns.to_list()
    cols2check = [x for x in cols if x not in ['Zone', 'Region', 'District', 'Year_FSNAU', 'CROP', 'Livelihood System']]
    df = df.dropna(subset=cols2check, how='all')

    df['FEWS use areaHarv'] = df[areaH_col] == df['Area']
    df.sort_values(['District', 'Year_FSNAU'], ascending=[True, True], inplace=True)
    df.to_csv(csv_in + 'merge4check_withFSNAU_Weston2025.csv', index=False)
    print('end')




