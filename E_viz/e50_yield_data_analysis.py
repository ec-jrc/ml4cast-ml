import pandas as pd
import numpy as np
import geopandas as gpd
import os
import matplotlib.pyplot as plt
from B_preprocess import b100_load

#
def barDfColumn(x, df, df_col_value, xticks, ylabel, crop_name, ax, sf_col_SD=None):
    if sf_col_SD == None:
        ax.bar(x, df[df_col_value].to_list())
    else:
        ax.bar(x, df[df_col_value].to_list(), yerr=df[sf_col_SD].to_list())
    xticks = [elem[:8] for elem in xticks]
    ax.set_xticks(x, xticks, rotation='vertical')
    ax.set_ylabel(ylabel)
    ax.set_title(crop_name)
    # plt.subplots_adjust(bottom=0.25)
    # plt.subplots_adjust(left=0.15)


def mapDfColumn(df, df_merge_col, df_col2map, df_col_admin_names, gdf, gdf_merge_col, gdf_gaul0_column, gdf_gaul0_name,
                lbl, cmap='tab20b', minmax=None, fn_fig=None, ax=None):
    """
    df: the data pandas df
    df_merge_col: the column name used to merge with geopandas df
    df_col2map: the column to map
    df_col_admin_names: the column with the admin names
    gdf: the geopandas df
    gdf_merge_col: the column name used to merge with pandas df
    gdf_gaul0_column: the gdf column with country name
    gdf_gaul0_name: the name of the gaul0 unit of interest (may be different from the one in the df)
    lbl: label for the legen
    fn_fig: full path to output figure
    """
    # join df with gdf
    gdf = gdf[gdf[gdf_gaul0_column] == gdf_gaul0_name]
    merged = gdf.merge(df, how='left', left_on=gdf_merge_col, right_on=df_merge_col)
    merged[df_col_admin_names] = merged[df_col_admin_names].fillna('')

    # Map it
    if fn_fig != None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    merged.boundary.plot(ax=ax, color="black", linewidth=0.5)
    vmin = merged[df_col2map].min()
    vmax = merged[df_col2map].max()
    if vmin == vmax:
        vmin = vmin - vmin/10
        vmax = vmax + vmax / 10
    if minmax != None:
        vmin = minmax[0]
        vmax = minmax[1]

    # https://matplotlib.org/stable/users/explain/colors/colormaps.html
    merged.plot(column=df_col2map, ax=ax, legend=True, legend_kwds={'label': lbl, 'orientation': "horizontal", 'shrink': 0.4},
                vmin=vmin, vmax=vmax, cmap=cmap) # tab20b
    # Add Labels (only plotted regions)
    #merged = merged[merged['Crop_name|first'] == c]
    merged['coords'] = merged['geometry'].apply(lambda x: x.representative_point().coords[:])
    merged['coords'] = [coords[0] for coords in merged['coords']]
    for idx, row in merged.iterrows():
        ax.annotate(text=row[df_col_admin_names], xy=row['coords'], horizontalalignment='center', fontsize=8, color='black')

    ax.set_xlabel('Deg E')
    start, end = ax.get_xlim()
    ss = np.floor(start)
    ee = np.ceil(end)
    ax.set_xlim(ss, ee)
    ax.set_xticks(np.arange(ss, ee, 1))
    # plt.xlim(start, end)

    ax.set_ylabel('Deg N')
    start, end = ax.get_ylim()
    ss = np.floor(start)
    ee = np.ceil(end)
    ax.set_ylim(ss, ee)
    ax.set_yticks(np.arange(ss, ee, 1))
    # plt.ylim(start, end)
    # plt.tight_layout()
    if fn_fig != None:
        fig.tight_layout()
        fig.savefig(fn_fig)  # , dpi = 300)
        plt.close(fig)



def mapYieldStats(config, fn_shape_gaul1, country_name_in_shp_file,  gdf_gaul0_column='name0', prct2retain=100):
    dir2use = os.path.join(config.data_dir, 'Label_analysis')
    # map numer of year available (Yield count)
    # map mean area, mean yield, trend?

    gdf_gaul1_id = "asap1_id"
    df_gaul1_id = "Region_ID|"
    gdf_gaul0_name = country_name_in_shp_file

    units = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_measurement_units.csv'))
    area_unit = units['Area'].values[0]
    yield_unit = units['Yield'].values[0]


    pd.set_option('display.max_columns', None)
    desired_width = 1000
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    #https://towardsdatascience.com/a-beginners-guide-to-create-a-cloropleth-map-in-python-using-geopandas-and-matplotlib-9cc4175ab630
    #load shp
    fp = fn_shape_gaul1
    gdf = gpd.read_file(fp)
    # load stats
    LTstats = pd.read_csv(os.path.join(dir2use, config.AOI + '_LTstats_retainPRCT' + str(prct2retain) + '.csv'))
    # LTstats.columns = LTstats.columns.map(lambda x: '|'.join([str(i) for i in x]))
    uniqueCrops = LTstats['Crop_name|first'].unique()

    # loop on crops
    for c in uniqueCrops:
        statsCrop = LTstats[LTstats['Crop_name|first'] == c]
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()
        # Yield|count (valid obs)
        lbl = c + ' Number of valida yield data'
        mapDfColumn(statsCrop, df_gaul1_id, 'Yield|count', 'AU_name|first', gdf, gdf_gaul1_id, gdf_gaul0_column,
                    gdf_gaul0_name, lbl, cmap='tab20b', fn_fig=None, ax=axs[0])
        # Percent crop area
        lbl = c + ' % of total crop area in the adm. unit'
        mapDfColumn(statsCrop, df_gaul1_id, 'Perc_area|', 'AU_name|first', gdf, gdf_gaul1_id, gdf_gaul0_column,
                    gdf_gaul0_name, lbl, cmap='YlGn', fn_fig=None, ax=axs[1], minmax=[0,100])
        # % national production
        lbl = c +' % national production'
        mapDfColumn(statsCrop, df_gaul1_id, 'Perc_production|', 'AU_name|first', gdf, gdf_gaul1_id, gdf_gaul0_column,
                    gdf_gaul0_name, lbl, cmap='YlGn', fn_fig=None, ax=axs[2], minmax=[0,100])
        # Total production
        if area_unit == 'ha' and yield_unit == 't/ha':
            divider = 1000
        elif area_unit == 'ha' and yield_unit == 'kg/ha':
            divider = 1000000
        else:
            print('Measurement units not foreseen')
            exit()
        statsCrop['Production|mean'] = statsCrop['Production|mean'] / divider
        lbl = c + ' production  [kt]'
        mapDfColumn(statsCrop, df_gaul1_id, 'Production|mean', 'AU_name|first', gdf, gdf_gaul1_id, gdf_gaul0_column, gdf_gaul0_name, lbl, cmap='YlGn', fn_fig=None, ax=axs[3])
        # Total area
        if area_unit == 'ha':
            divider = 100
        else:
            print('Measurement units not foreseen')
            exit()
        statsCrop['Area|mean'] = statsCrop['Area|mean'] / divider
        lbl = c + ' area [km^2]' #lbl = r'${\rm \/ Area \/ (1000 km^2)}$'
        mapDfColumn(statsCrop, df_gaul1_id, 'Area|mean', 'AU_name|first', gdf, gdf_gaul1_id, gdf_gaul0_column,
                    gdf_gaul0_name, lbl, cmap='YlGn', fn_fig=None, ax=axs[4])
        # yield
        lbl = c + ' Yield'
        mapDfColumn(statsCrop, df_gaul1_id, 'Yield|mean', 'AU_name|first', gdf, gdf_gaul1_id, gdf_gaul0_column,
                    gdf_gaul0_name, lbl, cmap='YlGn', fn_fig=None, ax=axs[5])

        fn_fig = os.path.join(dir2use, config.AOI + '_map_' + c + str(prct2retain) + '.png')
        fig.tight_layout()
        fig.savefig(fn_fig)  # , dpi = 300)
        plt.close(fig)


# def mapYieldStats(config, fn_shape_gaul1, country_name_in_shp_file,  gdf_gaul0_column='name0', prct2retain=100):
#     dir2use = os.path.join(config.data_dir, 'Analysis')
#     # map numer of year available (Yield count)
#     # map mean area, mean yield, trend?
#
#     gdf_gaul1_id = "asap1_id"
#     df_gaul1_id = "Region_ID|"
#     gdf_gaul0_name = country_name_in_shp_file
#
#     pd.set_option('display.max_columns', None)
#     desired_width = 1000
#     pd.set_option('display.width', desired_width)
#     np.set_printoptions(linewidth=desired_width)
#     #https://towardsdatascience.com/a-beginners-guide-to-create-a-cloropleth-map-in-python-using-geopandas-and-matplotlib-9cc4175ab630
#     #load shp
#     fp = fn_shape_gaul1
#     gdf = gpd.read_file(fp)
#     # load stats
#     LTstats = pd.read_csv(os.path.join(dir2use, config.AOI + '_LTstats_retainPRCT' + str(prct2retain) + '.csv'))
#     # LTstats.columns = LTstats.columns.map(lambda x: '|'.join([str(i) for i in x]))
#     uniqueCrops = LTstats['Crop_name|first'].unique()
#
#     # loop on crops
#     for c in uniqueCrops:
#         statsCrop = LTstats[LTstats['Crop_name|first'] == c]
#
#         # % national production
#         fn_fig = os.path.join(dir2use, config.AOI + '_map_' + c + str(prct2retain) + 'Perc_production' + '.png')
#         lbl = c +' % national production'
#         mapDfColumn(statsCrop, df_gaul1_id, 'Perc_production|', 'AU_name|first', gdf, gdf_gaul1_id, gdf_gaul0_column, gdf_gaul0_name, lbl, fn_fig)
#         # Yield|count (valid obs)
#         fn_fig = os.path.join(dir2use, config.AOI + '_map_' + c + str(prct2retain) + 'N_valid_years' + '.png')
#         lbl = c + ' Number of valida yield data'
#         mapDfColumn(statsCrop, df_gaul1_id, 'Yield|count', 'AU_name|first', gdf, gdf_gaul1_id, gdf_gaul0_column, gdf_gaul0_name, lbl, fn_fig)
#
#         # Percent crop area
#         fn_fig = os.path.join(dir2use, config.AOI + '_map_' + c + str(prct2retain) + 'Perc_area' + '.png')
#         lbl = c + ' % of total crop area'
#         mapDfColumn(statsCrop, df_gaul1_id, 'Perc_area|', 'AU_name|first', gdf, gdf_gaul1_id, gdf_gaul0_column,
#                     gdf_gaul0_name, lbl, fn_fig)
#
#         # Total area, skipped
#         lbl = r'${\rm \/ Area \/ (1000 km^2)}$'
#
#         # yield
#         fn_fig = os.path.join(dir2use, config.AOI + '_map_' + c + str(prct2retain) + 'yield' + '.png')
#         lbl = c + ' Yield'
#         mapDfColumn(statsCrop, df_gaul1_id, 'Yield|mean', 'AU_name|first', gdf, gdf_gaul1_id, gdf_gaul0_column,
#                     gdf_gaul0_name, lbl, fn_fig)