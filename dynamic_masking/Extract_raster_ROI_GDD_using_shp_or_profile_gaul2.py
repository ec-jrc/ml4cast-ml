import datetime as dt
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
import numpy.ma as ma


from rasterio.windows import Window
import asap_toolbox.util as tbx_util
import asap_toolbox.util.date
from rasterio.windows import from_bounds
from asap_toolbox.util import raster as tbx_raster
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
from dynamic_masking import gaussian_mixture
from matplotlib.dates import DateFormatter
import dm_local_constants as dm_cst
'''
This script computes the GDD per pixel at each dekad
It outputs images with requested ROI extent (based on shp or better on a template) of: NDVI, tav and GDD from 
requested time of year
'''


study = 'Ukraine'

##########################################################################
# SET INPUTS HERE
if study == 'Ukraine':
    AOI_FROM_TEMPLATE = dm_cst.TEMPLATE_OUTPUT_gaul2_Ukraine
    Temp_scale = 1
    Temp_offset = 0
    VALID_RANGE = [-100, 100]
    NDVI_scale = 0.0048
    NDVI_offset = -0.2000
    MIN_DN_FLAGS = 250
    COUNTRY = 'Ukraine'
    DIR_OUT_ROIS = dm_cst.DIR_OUT_ROIS_Ukraine_gaul2
    # base temperature for gdd computation
    TBASE = 0
    # time domain for GDD computation
    START_GDD_CUMULATION_MMDD = 1001
    # Output time series from to
    # START_YYYY = 2021 #2021
    # END_YYYYMMDD = 20220411 #20220321  # 20220311  ## 20151221
    # START_YYYY = 2020  # 2021
    # END_YYYYMMDD = 20210921
    # START_YYYY = 2019  # 2021
    # END_YYYYMMDD = 20200921
    START_YYYY = 2018  # 2021
    END_YYYYMMDD = 20190921
    START_YYYYMMDD = int(str(START_YYYY)+str(START_GDD_CUMULATION_MMDD))
    # GDD fixed grid parameters
    #DICT_GDD_GRID = {'MIN': 0, 'MAX': 500, 'N_STEP': 51}
    DICT_GDD_GRID = {'MIN': 0, 'MAX': 5000, 'N_STEP': 101}



def main():
    CHECK_FOLDER = os.path.isdir(DIR_OUT_ROIS)
    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(DIR_OUT_ROIS)
        print("created folder : ", DIR_OUT_ROIS)
    # set desire gdd grid
    gdd_fixed_grid = np.linspace(DICT_GDD_GRID['MIN'], DICT_GDD_GRID['MAX'], num=DICT_GDD_GRID['N_STEP'])
    # features = gpd.read_file(ADMIN_SHAPE)
    # features = features[features['adm0_name'] == COUNTRY]
    # # bounding box
    # bbox = features.bounds
    # if REGIONS[0] != 'all':
    #     features = features[features['adm1_name'].isin(REGIONS)]
    # define dekads list
    dekads_raw = range(tbx_util.date.string2raw(START_YYYYMMDD), tbx_util.date.string2raw(END_YYYYMMDD) + 1)
    dekads = [tbx_util.date.raw2dekade(dekade) for dekade in dekads_raw]
    # get dekKK of cumulation start
    dekKKstart = tbx_util.date.string2dekade(START_YYYYMMDD) % 100
    gdd = None
    fn_gdd_list = []
    fn_ndvi_list = []
    idx_processed_years = -1
    for dek in dekads:
        dekKK = dek % 100
        fn_in = os.path.normpath('//ies/d5/asap/ecmwf/io.5.0/tav/tav_{}.img').format(
            tbx_util.date.dekade2string(dek))
        #fn_in = os.path.normpath('//ies/d5/asap/asap.5.0/data/indicators_ndvi/ndvi/ndvi_{}.img').format(tbx_util.date.dekade2string(dek))
        fn_out = os.path.normpath(DIR_OUT_ROIS + '/tav_{}.tif').format(tbx_util.date.dekade2string(dek))
        fn_out_gdd = os.path.normpath(DIR_OUT_ROIS + '/gdd_{}.tif').format(tbx_util.date.dekade2string(dek))
        fn_out_ndvi = os.path.normpath(DIR_OUT_ROIS + '/ndvi_{}.tif').format(tbx_util.date.dekade2string(dek))

        if os.path.isfile(fn_out):
            print('Tav window of ', dek, ' already exists')
        else:
            print('Saving Tav window of ', dek)

            with rasterio.open(AOI_FROM_TEMPLATE) as src:
                template_bounds = src.bounds
                template_profile = src.profile
            with rasterio.open(fn_in) as src:
                win = rasterio.windows.from_bounds(template_bounds.left, template_bounds.bottom, template_bounds.right, template_bounds.top, transform=src.transform)
                data = src.read(1, window=win)
                #print(src)


            # create the masked Tav data set
            no_data_mask = np.logical_or(data < VALID_RANGE[0], data > VALID_RANGE[1])
            mdata = ma.masked_array(data, mask=no_data_mask)
            # convert to physical values
            ds_data = (mdata * Temp_scale) + Temp_offset
            # save the win
            # change dtype
            template_profile['dtype'] = 'float32'
            # Using same profile I get 1 km resolution out
            with rasterio.open(fn_out, 'w+',
                               **template_profile #, compress="DEFLATE"
                               ) as dst:
                dst.write(ma.filled(ds_data, fill_value=np.NaN), 1)
            # rested temp base
            ds_data[ds_data < TBASE] = 0
            nDaysInDek = float(tbx_util.date.get_dekade_days(tbx_util.date.dekade2date(dek)))
            if dekKK == dekKKstart:
                # we are at start
                gdd = ds_data * nDaysInDek
            else:
                # we passed start, keep on cumultaing
                gdd = np.add(gdd, ds_data * nDaysInDek)
            #save gdd
            with rasterio.open(fn_out_gdd, 'w+',
                               **template_profile  # , compress="DEFLATE"
                               ) as dst:
                dst.write(ma.filled(gdd, fill_value=np.NaN), 1)



           # Now NDVI same window
            fn_in = os.path.normpath('//ies/d5/asap/asap.5.0/data/indicators_ndvi/ndvi/ndvi_{}.img').format(
                tbx_util.date.dekade2string(dek))

            if os.path.isfile(fn_out_ndvi):
                print('NDVI window of ', dek, ' already exists')
            else:
                print('Saving NDVI window of ', dek)

                with rasterio.open(AOI_FROM_TEMPLATE) as src:
                    template_bounds = src.bounds
                    template_profile = src.profile
                with rasterio.open(fn_in) as src:
                    win = rasterio.windows.from_bounds(template_bounds.left, template_bounds.bottom,
                                                       template_bounds.right, template_bounds.top,
                                                       transform=src.transform)
                    data = src.read(1, window=win)
                        # print(src)

                # create the masked NDVI data set
                no_data_mask = data > MIN_DN_FLAGS
                mdata = ma.masked_array(data, mask=no_data_mask)
                # convert to physical values
                ds_data = (mdata * NDVI_scale) + NDVI_offset
                # save the win
                # change dtype
                template_profile['dtype'] = 'float32'
                with rasterio.open(fn_out_ndvi, 'w+',
                                   **template_profile #, compress="DEFLATE"
                                   ) as dst:
                    dst.write(ma.filled(ds_data, fill_value=np.NaN), 1)
        if dekKK == dekKKstart:
            fn_gdd_list.append([fn_out_gdd])
            fn_ndvi_list.append([fn_out_ndvi])
            idx_processed_years = idx_processed_years + 1
        else:
            fn_gdd_list[idx_processed_years].append(fn_out_gdd)
            fn_ndvi_list[idx_processed_years].append(fn_out_ndvi)

    #all inputs done, now rescale
    #all files, block by block
    # fn_gdd_list and fn_ndvi_list are list of list containing relevant files per GDD year

    #get template profile for output
    with rasterio.open(AOI_FROM_TEMPLATE) as src:
        template_profile = src.profile
        # change dtype
        template_profile['dtype'] = 'float32'
    # fn_gdd_list is a list of list, first level is the years, second level is the 36 files of that year
    for i in range(len(fn_gdd_list)):
        # prepare files for outupt
        template = os.path.splitext(fn_ndvi_list[i][0])[0].replace("ndvi","ndvi_xGDD") + r'_{}.tif'
        fn_out_list = []
        for g in gdd_fixed_grid.astype('int'):
            fn_out_list.append(os.path.normpath(template).format(str(f"{g:04d}")))
        # open all the filed for a year
        src_gdd_list = [rasterio.open(f) for f in fn_gdd_list[i]]
        src_ndvi_list = [rasterio.open(f) for f in fn_ndvi_list[i]]
        # open all to write
        ds_list = [rasterio.open(f, 'w+',**template_profile) for f in fn_out_list]
        # fetch the list of windows
        win_list = list(src_gdd_list[0].block_windows())
        # loop over window blocks
        for ji, window in win_list:
            #get the gdd and ndvi windoow
            gdd = np.concatenate([f.read(1, window=window) for f in src_gdd_list],axis=0)
            ndvi = np.concatenate([f.read(1, window=window) for f in src_ndvi_list])
            ndvi_gdd_axis = np.zeros((len(gdd_fixed_grid), ndvi.shape[1]))
            # loop on each single pixel
            for p in range(gdd.shape[1]):
                ndvi_gdd_axis[:,p] = np.interp(gdd_fixed_grid, gdd[:,p], ndvi[:,p], right=np.NaN)
            # write the windows
            for l, f in enumerate(ds_list):
                f.write(ndvi_gdd_axis[l, :].reshape(1, ndvi_gdd_axis[l,:].shape[0]), 1, window=window)
            # close the ouput_files
        [ds.close() for ds in ds_list]



    print('end')



if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    desired_width = 700
    pd.set_option('display.width', desired_width)
    start = dt.datetime.now()
    main()
    print('Execution time:', dt.datetime.now() - start)