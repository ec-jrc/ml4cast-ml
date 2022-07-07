import os
import rasterio
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import dm_local_constants as dm_cst


fn_info = dm_cst.fn_info


fn16 = dm_cst.fn16
fn16_warped = dm_cst.fn16_warped
fn17 = dm_cst.fn17
fn17_warped = dm_cst.fn17_warped
fn18 = dm_cst.fn18
fn19 = dm_cst.fn19
# gdalwarp_str = 'gdalwarp -r sum -overwrite -ot uint16 -t_srs EPSG:4326 -te {0} {1} {2} {3} -tr {4} {5} {6} {7}'
# gdalwarp_process = gdalwarp_str.format(xmin, ymin, xmax, ymax, resX, resY, mask_file, fn_resampled_mask)
# os.system(gdalwarp_process)
if False:
    with open(fn_info, 'w') as f:
        return_code = subprocess.call(['gdalinfo', '-norat', '-noct', fn16_warped], stdout=f)
        f.write('\n')
        # return_code = subprocess.call(['gdalinfo', '-norat', '-noct', fn17], stdout=f)
        # f.write('\n')
        # return_code = subprocess.call(['gdalinfo', '-norat', '-noct', fn18], stdout=f)
        # f.write('\n')
        # return_code = subprocess.call(['gdalinfo', '-norat', '-noct', fn19], stdout=f)
        # f.write('\n')
# I have to make 16 and 17 as 18 and 19 (so output extent is 19)
if False:
    run_cmd = [
        'gdalwarp',
        '-wo', 'INIT_DEST=NO_DATA',
        '-wo', 'EXTRA_ELTS=1',
        '-of', 'HFA',
        '-dstnodata', '-9999',
        # -te <xmin ymin xmax ymax>
        '-te', '77196.875', '6604343.855', '1000656.875', '7279043.855', #georeferenced extents of output file
        # -tr <xres> <yres>
        # Set output file resolution (in target georeferenced units).
        '-tr', '20', '-20',
        '-wt', 'Byte',
        '-ot', 'Byte',
        '-co', 'COMPRESSED=YES',
        fn17,
        fn17_warped,
    ]

    p = subprocess.run(run_cmd, shell=False, input='\n', capture_output=True, text=True)
    if p.returncode != 0:
        print(p)
# Classes:
# 1 Maize
# 2 (Planted) Pasture
# 3 Sunflower
# 4 SoyaBeans
# 5 Sorghum
# 6 Fallow
# 7 Groundnuts
# 8 No class
# 9 Wheat
# 10 WheatMaize
# 11 WheatSoya
# 23  Sugarcane
dict_clases = {1: 'Maize', 2: '(Planted) Pasture', 3: 'Sunflower', 4: 'SoyaBeans',
               5: 'Sorghum', 6: 'Fallow', 7: 'Groundnuts', 9: 'Wheat',
               10: 'WheatMaize', 11: 'WheatSoya', 23: 'Sugarcane'}

data = np.zeros((4, 33735, 46173), dtype=np.int8) #Size is 46173, 33735
filelist = [fn16_warped, fn17_warped, fn18, fn19]
for i, fn in enumerate(filelist):
    with rasterio.open(fn) as src:
        data[i,:,:] = src.read(1)
        crs = src.crs
        transform = src.transform
# I need to understand if pasture (an others) is permanet and then make a stable arable land map
if False:
    for k, v in dict_clases.items():
    # for cl in classes:
        sumOccurrence = np.sum(data==int(k), axis=0)
        #fn = dm_cst.output_path_stability+'\occurrence_class' + str(v) + '.tif'
        # with rasterio.open(fn, 'w+', driver='GTiff',
        #                    dtype=sumOccurrence.dtype,
        #                    height=sumOccurrence.shape[0],
        #                    width=sumOccurrence.shape[1],
        #                    count=1,
        #                    crs=crs,
        #                    transform=transform
        #                    ) as dst:
        #     dst.write(sumOccurrence, 1)
        # save a plot
        bins = [1,2,3,4,5]
        plt.hist(sumOccurrence.flatten(), bins=bins, density=False, edgecolor='black', linewidth=1, rwidth=0.5)
        plt.ylim(0, 6.5e7)
        plt.xlabel('Occurrence of the class over 4 years')
        plt.ylabel('Counts')
        plt.xticks([1.5,2.5,3.5,4.5], labels=[1,2,3,4])
        areakm2 = int(np.sum(sumOccurrence>0).astype('float') * 20 * 20 / 1000000)
        plt.title(v + ', area (n>1): ' + str(areakm2) + ' km2')
        fn =   dm_cst.output_path_stability+'\hhisto_occurrence_class_' + str(v) + '.png'
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()

# write the arable land
# never sugarcane, pasture no more than 1 times (2,3,4 exluded), fallow 4 time excluded
arable = np.zeros((33735, 46173), dtype=np.int8)
# set to 1 when one of the classes(exluding sugar cane) has at least occurred once
list_classes_no_sugar = [1,2,3,4,5,6,7,9,10,11]
sumOccurrence = np.sum(np.isin(data, list_classes_no_sugar), axis=0)
arable[sumOccurrence>0] = 1
# set to 0 where pasture is found 2,3, or for times
sumOccurrence = np.sum(data==2, axis=0)
arable[sumOccurrence>1] = 0
# set to 0 where fallow is always 4
sumOccurrence = np.sum(data==6, axis=0)
arable[sumOccurrence==4] = 0
# save
fn = dm_cst.fn_arable_outupt
with rasterio.open(fn, 'w+', driver='GTiff',
                       dtype=arable.dtype,
                       height=arable.shape[0],
                       width=arable.shape[1],
                       count=1,
                       crs=crs,
                       transform=transform) as dst:
    dst.write(arable, 1)


