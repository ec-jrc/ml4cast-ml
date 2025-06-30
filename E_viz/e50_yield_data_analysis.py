import pandas as pd
import numpy as np
import geopandas as gpd
import os
from pathlib import Path
import matplotlib.pyplot as plt
from B_preprocess import b101_load_cleaned
import pymannkendall as mk
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
#

pd.set_option('display.max_columns', None)
desired_width = 1000
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
def barDfColumn(x, df, df_col_value, xticks, ylabel, crop_name, ax, sf_col_SD=None):
    if sf_col_SD == None:
        ax.bar(x, df[df_col_value].to_list())
    elif np.isnan(df[sf_col_SD].to_list()).all():
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
                lbl, cmap='tab20b', minmax=None, fn_fig=None, ax=None, dropna=False):
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
    dropna: if True, drops rows with missing 'geometry' values
    """
    # join df with gdf
    gdf = gdf[gdf[gdf_gaul0_column] == gdf_gaul0_name]
    # merged = gdf.merge(df, how='left', left_on=gdf_merge_col, right_on=df_merge_col)
    merged = gdf.merge(df, how='right', left_on=gdf_merge_col, right_on=df_merge_col)
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
    if dropna:
        merged = merged.dropna(subset=['geometry'])
    merged['coords'] = merged['geometry'].apply(lambda x: x.representative_point().coords[:])
    merged['coords'] = [coords[0] for coords in merged['coords']]
    for idx, row in merged.iterrows():
        ax.annotate(text=row[df_col_admin_names], xy=row['coords'], horizontalalignment='center', fontsize=6, color='black')

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

def mapDfColumn2Ax(df, df_merge_col, df_col2map, df_col_admin_names, gdf, gdf_merge_col, gdf_gaul0_column, gdf_gaul0_name,
                lbl, cmap='tab20b', minmax=None, ax=None, cate=False):
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
    ax: ax to be filled
    """
    # join df with gdf
    gdf = gdf[gdf[gdf_gaul0_column] == gdf_gaul0_name]
    # merged = gdf.merge(df, how='left', left_on=gdf_merge_col, right_on=df_merge_col)
    merged = gdf.merge(df, how='right', left_on=gdf_merge_col, right_on=df_merge_col)
    merged[df_col_admin_names] = merged[df_col_admin_names].fillna('')

    merged.boundary.plot(ax=ax, color="black", linewidth=0.5)
    if cate == False:
        vmin = merged[df_col2map].min()
        vmax = merged[df_col2map].max()
        if vmin == vmax:
            vmin = vmin - vmin/10
            vmax = vmax + vmax / 10
        if minmax != None:
            vmin = minmax[0]
            vmax = minmax[1]

    # https://matplotlib.org/stable/users/explain/colors/colormaps.html
    if cate:
        merged = merged.dropna(subset=["colors"])
        # Create a list of patches representing the different categories
        patches = [mpatches.Patch(color=color, label=label) for color, label in zip(merged[['colors',df_col2map]].drop_duplicates()['colors'], merged[['colors',df_col2map]].drop_duplicates()[df_col2map])]
        merged.plot(column=df_col2map, ax=ax, legend=True,
                    # legend_kwds={'label': lbl, 'orientation': "horizontal", 'shrink': 0.3}, #legend_kwds={'label': lbl, 'bbox_to_anchor':(.5, 0.1),'fontsize':12,'frameon':False}, #'orientation': "horizontal",
                    categorical=True, color=merged["colors"])
        ax.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=len(merged["colors"]), title=lbl, frameon=False)

    else:
        merged.plot(column=df_col2map, ax=ax, legend=True, legend_kwds={'label': lbl, 'orientation': "horizontal", 'shrink': 0.8},
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
    return ax

def mapYieldStats(config, fn_shape_gaul1, country_name_in_shp_file,  gdf_gaul0_column='name0', adminID_column_name_in_shp_file = 'asap1_id', prct2retain=100):
    dir2use = os.path.join(config.data_dir, 'Label_analysis' + str(prct2retain))
    if not isinstance(config.fn_reference_shape, list):
    # if len(fn_shape_gaul1)==1:
        gdf_gaul1_id = adminID_column_name_in_shp_file
        df_gaul1_id = "adm_id|"
        gdf_gaul0_name = country_name_in_shp_file

        units = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_measurement_units.csv'))
        area_unit = units['Area'].values[0]
        yield_unit = units['Yield'].values[0]
        if 'Production' in units.columns:
            prod_units = units['Production'].values[0]
        else:
            prod_units = 'kt'
        #https://towardsdatascience.com/a-beginners-guide-to-create-a-cloropleth-map-in-python-using-geopandas-and-matplotlib-9cc4175ab630
        #load shp
        fp = fn_shape_gaul1
        gdf = gpd.read_file(fp)
        # load stats
        LTstats = pd.read_csv(os.path.join(dir2use, config.AOI + '_LTstats_retainPRCT' + str(prct2retain) + '.csv'))
        uniqueCrops = LTstats['Crop_name|first'].unique()

        # loop on crops
        for c in uniqueCrops:
            statsCrop = LTstats[LTstats['Crop_name|first'] == c]
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            axs = axs.flatten()
            # Yield|count (valid obs)
            lbl = c + ' Number of valida yield data'
            mapDfColumn(statsCrop, df_gaul1_id, 'Yield|count', 'adm_name|first', gdf, gdf_gaul1_id, gdf_gaul0_column,
                        gdf_gaul0_name, lbl, cmap='tab20b', fn_fig=None, ax=axs[0])
            # Percent crop area
            lbl = c + ' % of national crop area in the adm. unit'
            mapDfColumn(statsCrop, df_gaul1_id, 'Crop_perc_area|', 'adm_name|first', gdf, gdf_gaul1_id, gdf_gaul0_column,
                        gdf_gaul0_name, lbl, cmap='YlGn', fn_fig=None, ax=axs[1]) # , minmax=[0,100]
            # % national production
            lbl = c +' % national production'
            mapDfColumn(statsCrop, df_gaul1_id, 'Crop_perc_production|', 'adm_name|first', gdf, gdf_gaul1_id, gdf_gaul0_column,
                        gdf_gaul0_name, lbl, cmap='YlGn', fn_fig=None, ax=axs[2]) #, minmax=[0,100]
            # Total production
            if area_unit == 'ha' and yield_unit == 't/ha':
                divider = 1000
            elif area_unit == 'ha' and yield_unit == 'kg/ha':
                divider = 1000000
            elif prod_units == '100t':
                divider = 10
            else:
                print('Measurement units not foreseen')
                exit()
            statsCrop.loc[:,'Production|mean'] = statsCrop.loc[:,'Production|mean'] / divider
            lbl = c + ' production  [kt]'
            mapDfColumn(statsCrop, df_gaul1_id, 'Production|mean', 'adm_name|first', gdf, gdf_gaul1_id, gdf_gaul0_column, gdf_gaul0_name, lbl, cmap='YlGn', fn_fig=None, ax=axs[3])
            # Total area
            if area_unit == 'ha':
                divider = 100
            elif area_unit == 'km2':
                divider = 1
            else:
                print('Measurement units not foreseen')
                exit()
            statsCrop.loc[:, 'Area|mean'] = statsCrop.loc[:, 'Area|mean'] / divider
            lbl = c + ' area [km^2]' #lbl = r'${\rm \/ Area \/ (1000 km^2)}$'
            mapDfColumn(statsCrop, df_gaul1_id, 'Area|mean', 'adm_name|first', gdf, gdf_gaul1_id, gdf_gaul0_column,
                        gdf_gaul0_name, lbl, cmap='YlGn', fn_fig=None, ax=axs[4])
            # yield
            lbl = c + ' Yield'
            mapDfColumn(statsCrop, df_gaul1_id, 'Yield|mean', 'adm_name|first', gdf, gdf_gaul1_id, gdf_gaul0_column,
                        gdf_gaul0_name, lbl, cmap='YlGn', fn_fig=None, ax=axs[5])

            fn_fig = os.path.join(dir2use, config.AOI + '_map_' + c + str(prct2retain) + '.png')
            fig.tight_layout()
            fig.savefig(fn_fig)  # , dpi = 300)
            plt.close(fig)
    else:
        # Multiple shapefiles to be used (e.g. Morocco)
        gdf_gaul1_id = adminID_column_name_in_shp_file
        df_gaul1_id = adminID_column_name_in_shp_file
        gdf_gaul0_name = country_name_in_shp_file

        units = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_measurement_units.csv'))
        area_unit = units['Area'].values[0]
        yield_unit = units['Yield'].values[0]
        if 'Production' in units.columns:
            prod_units = units['Production'].values[0]
        else:
            prod_units = 'kt'

        # Loop through all shapefiles in fn_shape_gaul1
        for shp in fn_shape_gaul1:
            # Load shapefile
            fp = shp
            gdf = gpd.read_file(fp)

            # Load statistics
            LTstats = pd.read_csv(
                os.path.join(dir2use, config.AOI + '_LTstats_retainPRCT' + str(prct2retain) + '.csv'))

            # Load regions
            regNames = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_REGION_id.csv'))
            lookup_table = regNames[['adm_id', 'Adjusted_jrc_id_in_shp']].drop_duplicates()

            # Perform left join and overwrite LTstats
            LTstats = pd.merge(LTstats, lookup_table, left_on='adm_id|', right_on='adm_id', how='left')
            LTstats = LTstats.rename(columns={'Adjusted_jrc_id_in_shp': gdf_gaul1_id})

            uniqueCrops = LTstats['Crop_name|first'].unique()

            # Loop through all crops
            for c in uniqueCrops:
                statsCrop = LTstats[LTstats['Crop_name|first'] == c]
                fig, axs = plt.subplots(2, 3, figsize=(15, 10))
                axs = axs.flatten()

                # Yield|count (valid obs)
                lbl = c + ' Number of valid yield data'
                mapDfColumn(statsCrop, df_gaul1_id, 'Yield|count', 'adm_name|first', gdf, gdf_gaul1_id,
                            gdf_gaul0_column,
                            country_name_in_shp_file, lbl, cmap='tab20b', fn_fig=None, ax=axs[0], dropna=True)

                # Percent crop area
                lbl = c + ' % of national crop area in the adm. unit'
                mapDfColumn(statsCrop, df_gaul1_id, 'Crop_perc_area|', 'adm_name|first', gdf, gdf_gaul1_id,
                            gdf_gaul0_column,
                            country_name_in_shp_file, lbl, cmap='YlGn', fn_fig=None, ax=axs[1], minmax=[0, 100],
                            dropna=True)

                # % national production
                lbl = c + ' % national production'
                mapDfColumn(statsCrop, df_gaul1_id, 'Crop_perc_production|', 'adm_name|first', gdf, gdf_gaul1_id,
                            gdf_gaul0_column,
                            country_name_in_shp_file, lbl, cmap='YlGn', fn_fig=None, ax=axs[2], minmax=[0, 100],
                            dropna=True)

                # Total production
                if area_unit == 'ha' and yield_unit == 't/ha':
                    divider = 1000
                elif area_unit == 'ha' and yield_unit == 'kg/ha':
                    divider = 1000000
                elif prod_units == '100t':
                    divider = 10
                else:
                    print('Measurement units not foreseen')
                    exit()
                statsCrop.loc[:, 'Production|mean'] = statsCrop.loc[:, 'Production|mean'] / divider
                lbl = c + ' production [kt]'
                mapDfColumn(statsCrop, df_gaul1_id, 'Production|mean', 'adm_name|first', gdf, gdf_gaul1_id,
                            gdf_gaul0_column,
                            country_name_in_shp_file, lbl, cmap='YlGn', fn_fig=None, ax=axs[3],
                            dropna=True)

                # Total area
                if area_unit == 'ha':
                    divider = 100
                elif area_unit == 'km2':
                    divider = 1
                else:
                    print('Measurement units not foreseen')
                    exit()
                statsCrop.loc[:, 'Area|mean'] = statsCrop.loc[:, 'Area|mean'] / divider
                lbl = c + ' area [km^2]'
                mapDfColumn(statsCrop, df_gaul1_id, 'Area|mean', 'adm_name|first', gdf, gdf_gaul1_id,
                            gdf_gaul0_column,
                            country_name_in_shp_file, lbl, cmap='YlGn', fn_fig=None, ax=axs[4],
                            dropna=True)

                # Yield
                lbl = c + ' Yield'
                mapDfColumn(statsCrop, df_gaul1_id, 'Yield|mean', 'adm_name|first', gdf, gdf_gaul1_id,
                            gdf_gaul0_column,
                            country_name_in_shp_file, lbl, cmap='YlGn', fn_fig=None, ax=axs[5],
                            dropna=True)

                # Save figure for the current crop and shapefile
                fn_fig = os.path.join(dir2use, config.AOI + '_map_' + c + '_' + os.path.basename(shp) + str(
                    prct2retain) + '.png')
                fig.tight_layout()
                fig.savefig(fn_fig)
                plt.close(fig)
def trend_anlysis(config, prct2retain=100):
    # test significance
    alpha = 0.01

    outDir = os.path.join(config.data_dir, 'Label_analysis'+str(prct2retain))
    Path(outDir).mkdir(parents=True, exist_ok=True)

    x = b101_load_cleaned.LoadCleanedLabel(config)
    # regNames = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_REGION_id.csv'))
    # crop_name = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_CROP_id.csv'))
    units = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_measurement_units.csv'))
    area_unit = units['Area'].values[0]
    yield_unit = units['Yield'].values[0]
    if 'Production' in units.columns:
        prod_units = units['Production'].values[0]
    else:
        prod_units = 'kt'
    crops = x['Crop_name'].unique()
    for c in crops:
        xc = x[x['Crop_name'] == c]
        xMinMax = [xc['Year'].min(), xc['Year'].max()]
        yMinMax = [np.floor(xc['Yield'].min()), np.ceil(xc['Yield'].max())]
        adm = xc['adm_name'].unique()
        # fig, axs = plt.subplots(len(adm), 1, figsize=(10, 2.5 * len(adm)))
        fig, axs = plt.subplots(max([len(adm),2]), 1, figsize=(10, 2.5 * len(adm))) # the max here is used because there might be just one admin, and subplot does not return axes
        axs = axs.flatten()
        axs_counter = 0
        for a in adm:
            xca = xc[xc['adm_name'] == a].sort_values('Year')
            y = xca['Yield'].values.reshape(-1)
            X = xca['Year'].values.reshape(-1)
            # a minimum of two data is required
            n_valid = sum(~np.isnan(y))
            if n_valid >=2:
                trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(y, alpha=alpha)
                trend_line_mk = np.arange(len(y)) * slope + intercept
            else:
                trend = 'n < 2, trend cannot be assessed'
                p = np.NAN
            # axs[axs_counter].plot(X, y, label='Data') #, label=F'Theil-Sen trend line')
            axs[axs_counter].scatter(X, y, label='Data')  # , label=F'Theil-Sen trend line')
            axs[axs_counter].set_xlim(xMinMax)
            axs[axs_counter].set_ylim(yMinMax)
            axs[axs_counter].locator_params(integer=True)
            # ax = plt.gca()
            axs[axs_counter].grid(which='major', axis='x', linestyle='--')
            # Ensure x-axis tick marks are integers
            axs[axs_counter].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            if n_valid >= 2:
                if trend != 'no trend':
                    axs[axs_counter].plot(X, trend_line_mk, label='Theil-Sen trend line')
            axs[axs_counter].set_xlabel('Years')
            axs[axs_counter].set_ylabel('Yield [' + yield_unit + ']')
            axs[axs_counter].set_title(c + ', ' + a + ', ' + trend + '(p=' + str(np.round(p, 4))+ ')')
            axs[axs_counter].legend(frameon=False, loc='upper left')
            axs_counter = axs_counter + 1
        # save the fig
        fn_out = outDir + '/trend_analysis_' + c + '.pdf'
        fig.tight_layout()
        plt.savefig(fn_out)#, bbox_inches='tight')
        plt.close()
    print('end')
