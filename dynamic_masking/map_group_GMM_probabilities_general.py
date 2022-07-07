import datetime as dt
import pandas as pd
import numpy as np
import numpy.ma as ma
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds

from rasterio.windows import Window
import asap_toolbox.util as tbx_util
import asap_toolbox.util.date
from asap_toolbox.util import raster as tbx_raster
import matplotlib.pyplot as plt
import os
import dm_local_constants as dm_cst

"""
This script loads the probabilities from gam model (by asap id) and classify corresponding NDVI images
Besides the P of pixel being class 1 (the one with largest mean) and class 2 (the class with lower mean),
it saves AFI, admin raster and all used NDVI files in the ouput files
"""

study = 'Ukraine'

#####################################
# SET INPUTS HERE
if study == 'test':
    print('Study not implemented yet')
elif study == 'Ukraine':
    # Set the AFI mask used to compute GMM (int 0-200 expected here)
    CROP_MASK_F = dm_cst.afi_CROP_MASK_F_Ukraine_2019
    CROP_TYPE_NAME = ['wheat_2019']
    ADMIN_SHAPE = dm_cst.ADMIN_SHAPE_Ukraine_gaul1
    NDVI_PATH = dm_cst.NDVI_PATH
    NDVI_scale = 0.0048
    NDVI_offset = -0.2000
    CROP_MASK_SCALE = 0.005
    # dir where GMM responsibilities are stored
    DIR_OUT_FIGS = dm_cst.DIR_OUT_FIGS_map_group_GMM_probabilities_general
    FN_RESPONSABILITY = dm_cst.FN_RESPONSABILITY_map_group_GMM_probabilities_general
    # out results
    DIR_OUT_MAPS = dm_cst.DIR_OUT_MAPS_map_group_GMM_probabilities_general
    # Date used to retrieve resonsibilities that are applied to all dates listed in the responsability file
    MMDDRESP = [3, 11] #[month, day]
    #PERIOD_WINTER = [9, 1]  # [month, day]'09-01'
    #PERIOD_SUMMER = [2, 1]  # '02-21'
    CLASSES_NAMES = ['class1_highNDVI', 'class2_lowNDVI']

OUTPUT_FLAG_ONE_GAUSSIAN = 100
OUTPUT_FLAG_NO_AFI = 110
OUTPUT_FLAG_INVALID_NDVI = 120
#####################################

def main():
    CHECK_FOLDER = os.path.isdir(DIR_OUT_MAPS)
    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(DIR_OUT_MAPS)
        print("created folder : ", DIR_OUT_MAPS)

    df = pd.read_pickle(FN_RESPONSABILITY)
    # get the country
    country = df['adm0_name'].unique()[0]
    # get the dates to be processed
    dates = df['Date'].unique()

    # read the admin file and make a raster fo the country, do the same for AFI
    features = gpd.read_file(ADMIN_SHAPE)
    features = features[features['adm0_name'] == country]
    # bounding box
    bbox = features.bounds
    # get crs calculate window
    with rasterio.open(CROP_MASK_F) as src:
        afi_data = src.read(1)
        afi_bounds = src.bounds
        afi_profile = src.profile

    #
    # with rasterio.open(CROP_MASK_F) as src:
    #     # window=from_bounds(left, bottom, right, top, src.transform),
    #     win = from_bounds(bbox.minx.min(), bbox.miny.min(), bbox.maxx.max(), bbox.maxy.max(), src.transform)
    #     afi_data = src.read(1, window=win)
    #     win_transform = src.window_transform(win)
    #     crs = src.crs
    # save the afi window
    fn = os.path.join(DIR_OUT_MAPS, 'afi_raster.tif')
    afi_data = afi_data.astype('float64')*CROP_MASK_SCALE

    # change dtype
    afi_profile['dtype'] = 'float32'
    with rasterio.open(fn, 'w+',
                       **afi_profile
                       ) as dst:
        dst.write(afi_data, 1)

    # with rasterio.open(fn, 'w+', driver='GTiff',
    #                    height=win.height,
    #                    width=win.width,
    #                    count=1,
    #                    dtype=afi_data.dtype,
    #                    crs=crs,
    #                    transform=win_transform,
    #                    ) as dst:
    #     dst.write(afi_data, 1)

    # save the admin raster window
    fn = os.path.join(DIR_OUT_MAPS, 'admin_raster.tif')
    # change dtype
    afi_profile['dtype'] = 'int16'
    with rasterio.open(fn, 'w+',
                       **afi_profile
                       ) as out:
    # with rasterio.open(fn, 'w+', driver='GTiff',
    #                    height=win.height,
    #                    width=win.width,
    #                    count=1,
    #                    dtype='int16',
    #                    window=from_bounds(bbox.minx.min(), bbox.miny.min(), bbox.maxx.max(), bbox.maxy.min(), src.transform), #window=from_bounds(left, bottom, right, top, src.transform),
    #                    crs=crs,
    #                    transform=win_transform) as out:
        out_arr = out.read(1)
        # this is where we create a generator of geom, value pairs to use in rasterizing
        shapes = ((geom, value) for geom, value in zip(features.geometry, features.asap1_id))
        adminIDburned = rasterio.features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
        out.write_band(1, adminIDburned)
    # loop on NDVI file and get the win, save for further analysis
    dates = df.Date.unique()
    for date in dates:
        dek = tbx_util.date.string2dekade(date.strftime('%Y%m%d'))
        fn_out = os.path.normpath(DIR_OUT_MAPS + '/ndvi_{}.tif').format(tbx_util.date.dekade2string(dek))
        if os.path.isfile(fn_out):
            print('NDVI window of ', date, ' already exists')
        else:
            print('Saving NDVI window of ', date)
            f_path = os.path.normpath(NDVI_PATH + '/ndvi_{}.img').format(tbx_util.date.dekade2string(dek))
            with rasterio.open(f_path) as src:
                win = rasterio.windows.from_bounds(afi_bounds.left, afi_bounds.bottom, afi_bounds.right,
                                                   afi_bounds.top, transform=src.transform)
                data = src.read(1, window=win)
            # with rasterio.open(f_path) as src:
            #     win0 = from_bounds(bbox.minx.min(), bbox.miny.min(), bbox.maxx.max(), bbox.maxy.max(), src.transform)
            #     data = src.read(1, window=win0)
            #     win_transform = src.window_transform(win0)
            #     crs = src.crs

                # create the masked NDVI data set
            no_data_mask = data > 250
            mdata = ma.masked_array(data, mask=no_data_mask)
            # convert to physical values
            ds_data = (mdata * NDVI_scale) + NDVI_offset
            # save the win
            # change dtype
            afi_profile['dtype'] = 'float32'
            with rasterio.open(fn_out, 'w+',
                               **afi_profile
                               ) as dst:
                dst.write(ma.filled(ds_data, fill_value=np.NaN), 1)
            # with rasterio.open(fn_out, 'w+', driver='GTiff',
            #                    height=win0.height,
            #                    width=win0.width,
            #                    count=1,
            #                    dtype=ds_data.dtype,
            #                    crs=crs,
            #                    transform=win_transform,
            #                    ) as dst:
            #     dst.write(ma.filled(ds_data, fill_value=np.NaN), 1)
    groups = CLASSES_NAMES

    #groups = ['winter', 'summer']
    #groupDates = [dt.date(dates[0].year, PERIOD_WINTER[0], PERIOD_WINTER[1]), dt.date(dates[0].year, PERIOD_SUMMER[0], PERIOD_SUMMER[1])]
    respDate = dt.date(dates[0].year, MMDDRESP[0], MMDDRESP[1])

    for i, group in enumerate(CLASSES_NAMES):
        print(group, ', date from which responsability is taken: ', respDate)
        df_date = df.loc[df['Date'] == respDate]
        # get NDVI of relevant date
        dek = tbx_util.date.string2dekade(respDate.strftime('%Y%m%d'))
        f_path = os.path.normpath(DIR_OUT_MAPS + '/ndvi_{}.tif').format(tbx_util.date.dekade2string(dek))
        with rasterio.open(f_path) as src:
            ndvi = src.read(1)
            #win_transform = src.window_transform(win)
        no_data_mask = np.isnan(ndvi)
        # prepare a masked empty array for output
        p_data = np.copy(ndvi)*0+OUTPUT_FLAG_ONE_GAUSSIAN # fill with just one
        mask_no_afi = afi_data == 0
        # loop on admin units
        for index, row in df_date.iterrows():
            print(row.adm1_name)
            rasterID = row.asap1_id
            # the gaussian with larger mean (the first) is the one of group
            # check that there are two gaussian
            if row.responsibilities.shape[1] == 2:
                for b in range(0, len(row.bins) - 1):
                    # print(row.bins[i])
                    # last bin is closed interval
                    if b == len(row.bins) - 1:
                        p_data[np.where((ndvi >= row.bins[b]) & (ndvi <= row.bins[b + 1]) & (adminIDburned == rasterID))] = row.responsibilities[b, i]
                    else:
                        p_data[np.where((ndvi >= row.bins[b]) & (ndvi < row.bins[b + 1]) & (adminIDburned == rasterID))] = row.responsibilities[b, i]
        p_data = ma.masked_array(p_data, mask=no_data_mask)
        p_data.data[mask_no_afi] = OUTPUT_FLAG_NO_AFI
        fn = os.path.normpath(DIR_OUT_MAPS + '/P_of_' + group + '_from_' + str(dek) + '.tif')
        # change dtype
        afi_profile['dtype'] = p_data.dtype
        with rasterio.open(fn, 'w+',
                           **afi_profile
                           ) as dst:
            dst.write(ma.filled(p_data, fill_value=OUTPUT_FLAG_INVALID_NDVI), 1)
        # with rasterio.open(fn, 'w+', driver='GTiff',
        #                    height=win.height,
        #                    width=win.width,
        #                    count=1,
        #                    dtype=p_data.dtype,
        #                    crs=crs,
        #                    transform=win_transform,
        #                    ) as dst:
        #     dst.write(ma.filled(p_data, fill_value=OUTPUT_FLAG_INVALID_NDVI), 1)





    print('end')


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    desired_width = 700
    pd.set_option('display.width', desired_width)
    start = dt.datetime.now()
    main()
    print('Execution time:', dt.datetime.now() - start)
