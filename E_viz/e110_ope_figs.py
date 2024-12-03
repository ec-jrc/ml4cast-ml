from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import pathlib
import calendar
from D_modelling import d090_model_wrapper, d140_modelStats
from D_modelling import d140_modelStats
from E_viz import e50_yield_data_analysis


def map(b1, config, var4time, OutputDir, fn_shape_gaul1, country_name_in_shp_file,  gdf_gaul0_column='name0', title='', suffix=''): #onfig, fn_shape_gaul1, country_name_in_shp_file,  gdf_gaul0_column='name0'
    #b1 is the df containing the consolidated forecasts
    df_regNames = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_REGION_id.csv'))
    crops = b1['Crop_name'].unique()
    #forcTimes = b1[var4time].unique()
    fp = fn_shape_gaul1
    gdf = gpd.read_file(fp)
    gdf_gaul1_id = "asap1_id"
    gdf_gaul0_column = 'name0'
    for c in crops:
        df_c = b1[(b1['Crop_name'] == c)].copy()
        # statsByAdmin = df_c.merge(df_regNames, how='left', left_on='adm_id', right_on='adm_id')
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))
        axs = axs.flatten()
        fig_name = OutputDir + '/' + datetime.today().strftime('%Y%m%d') + '_' + config.country_name_in_shp_file + '_' + c + '_AU_forecasts' + suffix + '.png'
        # plot production

        lbl = 'Yield forecast'
        # def min max and color table
        e50_yield_data_analysis.mapDfColumn(df_c, 'adm_id', 'fyield', 'Region_name', gdf, gdf_gaul1_id, gdf_gaul0_column,
                    country_name_in_shp_file, lbl, cmap='tab20b', fn_fig=None, ax=axs[0])
        lbl = "YF % difference with last avail. 5 years"
        minmax = [-df_c['fyield_diff_pct (last 5 yrs in data avail)'].abs().max(), df_c['fyield_diff_pct (last 5 yrs in data avail)'].abs().max()]
        e50_yield_data_analysis.mapDfColumn(df_c, 'adm_id', 'fyield_diff_pct (last 5 yrs in data avail)', 'Region_name', gdf, gdf_gaul1_id,
                    gdf_gaul0_column, country_name_in_shp_file, lbl, cmap='bwr_r', fn_fig=None, ax=axs[1], minmax=minmax)
        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        fig.savefig(fig_name)
        plt.close(fig)
