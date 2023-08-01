import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import dynamic_masking.dm_local_constants as dm_cst


""""
Take a UTM map, reproject to lat lon compatible with NDVI and create AFIs
Implemented studies:
'SAwest' South Africa west map (Wetsren Cape) 
'ZAeast' South Africa east map (4 provinces)
'Ukraine'
"""
study = 'ZAeast' #'Ukraine'#'ZAwest'

gdal_path = dm_cst.gdal_path
gdal_calc_path = os.path.join(gdal_path, 'gdal_calc.py')
gdal_path_apps= dm_cst.gdal_path_apps
gdal_warp_path = os.path.join(gdal_path, 'gdalwarp')
#km2 of NDVI pixels (rough estimate)
pix2km2 = 0.860946376547824
if study == 'steffen':
    TEMPLATE_OUTPUT = dm_cst.TEMPLATE_OUTPUT_ZAeast
    # DICT_AFI = {'summer_crops_2018': [1, 3, 4, 5, 7],
    #             'winter_crops_2018': [9],
    #             'win2sum_crops_2018': [10, 11],
    #             'pastures_2018': [2],
    #             'fallow_2018': [6],
    #             'arable_2018': [1, 3, 4, 5, 6, 7, 9, 10, 11],
    #             'sugarcane_2018': [23]}
    DICT_AFI = {'arable_land': [1]}
    input_file = dm_cst.input_file_ZAeast
    output_path = dm_cst.output_path_ZAeast
if study == 'ZAeast':
    TEMPLATE_OUTPUT = dm_cst.TEMPLATE_OUTPUT_ZAeast
    # DICT_AFI = {'summer_crops_2018': [1, 3, 4, 5, 7],
    #             'winter_crops_2018': [9],
    #             'win2sum_crops_2018': [10, 11],
    #             'pastures_2018': [2],
    #             'fallow_2018': [6],
    #             'arable_2018': [1, 3, 4, 5, 6, 7, 9, 10, 11],
    #             'sugarcane_2018': [23]}
    DICT_AFI = {'arable_land': [1]}
    input_file = dm_cst.input_file_ZAeast
    output_path = dm_cst.output_path_ZAeast
elif study == 'ZAwest':
    TEMPLATE_OUTPUT = dm_cst.TEMPLATE_OUTPUT_ZAwest
    DICT_AFI = {'winter_crops_2018': [9, 15, 16, 17, 19],
                'pastures_2018': [2],
                'fallow_2018': [6],
                'wheat_2018': [9],
                'barley_2018': [15],
                'canola_2018': [16],
                'oats_2018': [17],
                'lupine_2018': [19],
                'arable_2018': [6, 9, 15, 16, 17, 19],
                'Rooibos': [24]}
    input_file = dm_cst.input_file_ZAwest
    output_path = dm_cst.output_path_ZAwest
elif study == 'Ukraine':
    TEMPLATE_OUTPUT = dm_cst.TEMPLATE_OUTPUT_Ukraine
    #DICT_AFI = {'wheat_2019': [2],
    #            'maize_2019': [5]}
    DICT_AFI = {'arable_2019': [2,3,4,5,6,7,8,9,15,16,17]}
    #DICT_AFI = {'winter_2019': [2, 3]}
    #             #'spring_2019': [4],
    #             #'summer_2019': [5, 6, 7, 8, 16]}
    input_file = dm_cst.input_file_Ukraine_2019
    output_path = dm_cst.output_path_Ukraine

    # DICT_AFI = {'winter_2020': [2, 3]}
    # input_file = dm_cst.input_file_Ukraine_2020

    # DICT_AFI = {'winter_2021': [2, 3]}
    # input_file = dm_cst.input_file_Ukraine_2021
    # 'spring_2019': [4],
    # 'summer_2019': [5, 6, 7, 8, 16]}

    # DICT_AFI = {'winter_2016': [2, 3, 9, 15],
    #             'spring_2016': [4],
    #             'summer_2016': [5, 6, 7, 8, 16]}
    # input_file = dm_cst.input_file_Ukraine_2016
    output_path = dm_cst.output_path_Ukraine

afis = DICT_AFI.keys()

# get info from template output (it defines spatial extent and resolution of the AFI based on NDVI grid)
template = rasterio.open(TEMPLATE_OUTPUT)
# print(template.bounds)
xmin = template.bounds[0]
ymin =template.bounds[1]
xmax =template.bounds[2]
ymax =template.bounds[3]
resX = template.transform[0]
resY = template.transform[4]
epsg = template.crs.to_epsg()


for afi in afis:
    classes = DICT_AFI[afi]
    mask_file = os.path.join(output_path, 'mask.tif')
    # mask with all ones used to determine the area of the low res pixel
    control_mask_file = os.path.join(output_path, 'control_mask.tif')
    afi_float_file = os.path.join(output_path, 'afi_'+afi+'_float.tif')
    afi_int_file = os.path.join(output_path, 'afi_'+afi+'_int.tif')
    fn_resampled_mask = os.path.join(os.path.dirname(mask_file), 'resampled_' +os.path.basename(mask_file))
    fn_resampled_control_mask = os.path.join(os.path.dirname(control_mask_file), 'resampled_' +os.path.basename(control_mask_file))

    if True:
        # Creat binary mask
        # calc_expr = '"numpy.isin(A, [1, 3, 4, 5, 7])"'
        calc_expr = '"numpy.isin(A, ' + str(classes) + ')"'
        print(calc_expr)
        # Generate string of process.
        gdal_calc_str = 'python {0} -A {1} --outfile={2} --calc={3} --overwrite'
        gdal_calc_process = gdal_calc_str.format(gdal_calc_path, input_file,
        mask_file, calc_expr)
        # Call process.
        os.system(gdal_calc_process)

        # create control mask all 1
        calc_expr = '"numpy.ones_like(A)"'
        gdal_calc_process = gdal_calc_str.format(gdal_calc_path, input_file,
        control_mask_file, calc_expr)
        os.system(gdal_calc_process)

        gdalwarp_str = 'gdalwarp -r sum -overwrite -ot uint16 -t_srs EPSG:4326 -te {0} {1} {2} {3} -tr {4} {5} {6} {7}'
        gdalwarp_process = gdalwarp_str.format(xmin, ymin, xmax, ymax, resX, resY, mask_file, fn_resampled_mask)
        os.system(gdalwarp_process)

        #now control mask
        gdalwarp_process = gdalwarp_str.format(xmin, ymin, xmax, ymax, resX, resY, control_mask_file, fn_resampled_control_mask)
        os.system(gdalwarp_process)

    # now compute afi
    # DO ALL WITH RASTERIO..
    with rasterio.open(os.path.join(fn_resampled_mask)) as src:
        A = src.read(1)
        crs = src.crs
    with rasterio.open(os.path.join(fn_resampled_control_mask)) as src:
        B = src.read(1)
        crs = src.crs
        transform = src.transform
    # value 255 is for Nan
    maskNaN = A == 255
    res = np.array(A).astype(np.float32) / np.array(B).astype(np.float32)
    res[maskNaN] = 0

    with rasterio.open(afi_float_file, 'w+', driver='GTiff',
                       dtype=res.dtype,
                       height=res.shape[0],
                       width=res.shape[1],
                       count=1,
                       crs=crs,
                       transform=transform
                       ) as dst:
        dst.write(res, 1)

    resint = np.around(res*200, decimals=0).astype(np.uint8)
    with rasterio.open(afi_int_file, 'w+', driver='GTiff',
                       dtype=resint.dtype,
                       height=resint.shape[0],
                       width=resint.shape[1],
                       count=1,
                       crs=crs,
                       transform=transform
                       ) as dst:
        dst.write(resint, 1)
    #plot a frequency histo (n by bins of afi)
    nbins = 50
    bins = np.linspace(0, 100, nbins, endpoint=True)
    # exclude 0
    bins[0] = 0.0000001
    res100 = res / 2
    res100 = res100[res100 != 0]
    areak2m = np.sum(res100/100*pix2km2)

    plt.hist(res100, bins=bins, density=False) #, histtype='stepfilled', alpha=0.4)
    plt.xlabel('AFI>0')
    plt.ylabel('Counts')
    plt.title(afi + ', Area =' + str(areak2m) + ' km2')
    plt.savefig(os.path.join(output_path, afi + '_histo.png'))
    plt.close()

    print('ended ' + afi)





# OLD STUF
# calc_expr = '"A.astype(numpy.float32)/B"'
# gdal_calc_str = 'python {0} -A {1} -B {2} --type=Float32 --overwrite --outfile={3} --calc={4}'
# gdal_calc_process = gdal_calc_str.format(gdal_calc_path, fn_resampled_mask, fn_resampled_control_mask, afi_float_file, calc_expr)
# os.system(gdal_calc_process)
# # scaled to Int
# calc_expr = '"((A.astype(numpy.float32)/B)*200)"' #[A==255]=0
# gdal_calc_str = 'python {0} -A {1} -B {2} --type=Byte --overwrite --NoDataValue=0 --hideNoData --outfile={3} --calc={4}'
# gdal_calc_process = gdal_calc_str.format(gdal_calc_path, fn_resampled_mask, fn_resampled_control_mask, output_afi_path_tmp, calc_expr)
# os.system(gdal_calc_process)
# #set 255 to 0
# #calc_expr = '"numpy.where(A!=255, A, 0)"' #[A==255]=0
# gdal_calc_str = 'python {0} -A {1} --type=Byte --overwrite --outfile={2} --calc={3}'
# gdal_calc_process = gdal_calc_str.format(gdal_calc_path, output_afi_path_tmp, afi_int_file, calc_expr)
# #os.system(gdal_calc_process)
#
# # # Calc percentage
# # # raw values
# # gdal_calc - A
# # F:\tmp\AFI_CALC\ver2\crop_mask_resampled.tif - B
# # F:\tmp\AFI_CALC\ver2\control_mask_resampled.tif - -type = Float32 - -outfile = F:\tmp\AFI_CALC\ver2\AFI_RAW.tif - -calc = "A.astype(numpy.float32)/B"
# # # scaled to Int
# # gdal_calc - A
# # F:\tmp\AFI_CALC\ver2\crop_mask_resampled.tif - B
# # F:\tmp\AFI_CALC\ver2\control_mask_resampled.tif - -type = Byte - -outfile = F:\tmp\AFI_CALC\ver2\AFI_OUT.tif - -calc = "(A.astype(numpy.float32)/B)*200"