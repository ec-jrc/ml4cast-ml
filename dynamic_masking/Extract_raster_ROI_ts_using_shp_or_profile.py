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


study = 'Ukraine'# 'east'#'west'

##########################################################################
# SET INPUTS HERE
if study == 'Ukraine':
    ADMIN_SHAPE = dm_cst.ADMIN_SHAPE_Ukraine_gaul1
    AOI_FROM_TEMPLATE = dm_cst.AOI_FROM_TEMPLATE_Ukraine
    NDVI_PATH = '//ies/d5/asap/asap.5.0/data/indicators_ndvi/ndvi'
    NDVI_scale = 0.0048
    NDVI_offset = -0.2000
    MIN_DN_FLAGS = 250
    REGIONS = ['all'] #['all'] #["Khersons'ka"] # ['all'] #["Cherkas'ka"] #['all']
    COUNTRY = 'Ukraine'
    DIR_OUT_ROIS = dm_cst.DIR_OUT_ROIS_ts__Ukraine
    # time domain for the time profile
    START_YYYYMMDD = 20181001  ## 20150101
    END_YYYYMMDD = 20190921 #20220311  ## 20151221

def main():
    CHECK_FOLDER = os.path.isdir(DIR_OUT_ROIS)
    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(DIR_OUT_ROIS)
        print("created folder : ", DIR_OUT_ROIS)
    features = gpd.read_file(ADMIN_SHAPE)
    features = features[features['adm0_name'] == COUNTRY]
    # bounding box
    bbox = features.bounds
    if REGIONS[0] != 'all':
        features = features[features['adm1_name'].isin(REGIONS)]
        # define dekads list
    dekads_raw = range(tbx_util.date.string2raw(START_YYYYMMDD), tbx_util.date.string2raw(END_YYYYMMDD) + 1)
    dekads = [tbx_util.date.raw2dekade(dekade) for dekade in dekads_raw]
    for dek in dekads:
        fn_in = os.path.normpath('//ies/d5/asap/asap.5.0/data/indicators_ndvi/ndvi/ndvi_{}.img').format(tbx_util.date.dekade2string(dek))
        fn_out = os.path.normpath(DIR_OUT_ROIS + '/ndvi_{}.tif').format(tbx_util.date.dekade2string(dek))
        if os.path.isfile(fn_out):
            print('NDVI window of ', dek, ' already exists')
        else:
            print('Saving NDVI window of ', dek)
            if AOI_FROM_TEMPLATE == '':
                with rasterio.open(fn_in) as src:
                    win0 = from_bounds(bbox.minx.min(), bbox.miny.min(), bbox.maxx.max(), bbox.maxy.max(), src.transform)
                    win0 = win0.round_lengths()
                    win0 = win0.round_offsets()
                    tmp = win0.todict()
                    tmp['height'] += 2
                    tmp['width'] += 2
                    win0 = Window(**tmp)
                    data = src.read(1, window=win0)
                    win_transform = src.window_transform(win0)
                    crs = src.crs
            else:
                with rasterio.open(AOI_FROM_TEMPLATE) as src:
                    template_bounds = src.bounds
                    template_profile = src.profile
                with rasterio.open(fn_in) as src:
                    win = rasterio.windows.from_bounds(template_bounds.left, template_bounds.bottom, template_bounds.right, template_bounds.top, transform=src.transform)
                    data = src.read(1, window=win)
                    #print(src)


            # create the masked NDVI data set
            no_data_mask = data > MIN_DN_FLAGS
            mdata = ma.masked_array(data, mask=no_data_mask)
            # convert to physical values
            ds_data = (mdata * NDVI_scale) + NDVI_offset
            # save the win
            # change dtype
            template_profile['dtype'] = 'float32'
            with rasterio.open(fn_out, 'w+',
                               **template_profile
                               ) as dst:
                dst.write(ma.filled(ds_data, fill_value=np.NaN), 1)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    desired_width = 700
    pd.set_option('display.width', desired_width)
    start = dt.datetime.now()
    main()
    print('Execution time:', dt.datetime.now() - start)