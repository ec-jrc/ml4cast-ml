import datetime as dt
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio


from rasterio.windows import Window
import asap_toolbox.util as tbx_util
import asap_toolbox.util.date
from asap_toolbox.util import raster as tbx_raster
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import glob
import re
from dynamic_masking import gaussian_mixture
from matplotlib.dates import DateFormatter
import dm_local_constants as dm_cst


'''
The script works with any crop type AFI mask and with general cropmask AFI
Set desired item with
MASK_TYPE = 'crop_type' or 'generic'
This script loops over all required admin units of a country and retrieve AFI weighted NDVI temporal profile for a given time period of
selected crop types.
If MASK_TYPE is 'generic' it also compute Gaussian Mixture Model GMM and save responsibilities for crop group mapping from NDVI 
That is, dekad by dekad, compute Gaussian Mixture Model and plot results

Inputa data
- admin units
- crop type AFIs
- NDVI time series
Implemented studies:
'SAwest' South Africa west map (Wetsren Cape) 
'SAeast' South Africa east map (4 provinces)
'''

study = 'Ukraine'# 'east'#'west'

##########################################################################
# SET INPUTS HERE

if study == 'Ukraine':
    ADMIN_SHAPE = dm_cst.ADMIN_SHAPE_Ukraine_gaul1
    # dictionary can be taken from AFI_maker.py
    # DICT_AFI = {'wheat_2019': [2],
    #             'maize_2019': [5]}
    # DICT_AFI = {'winter_2019': [2, 3, 9, 15],
    #             'spring_2019': [4],
    #             'summer_2019': [5, 6, 7, 8, 16]}
    # DICT_AFI = {'winter_2016': [2, 3, 9, 15],
    #             'spring_2016': [4],
    #             'summer_2016': [5, 6, 7, 8, 16]}
    REGIONS = ['all'] #['all'] #["Khersons'ka"] # ['all'] #["Cherkas'ka"] #['all']
    DICT_AFI = {'winter_2019': [2, 3, 15]}
    REGIONS = ["Khersons'ka"]
    dir_afis = dm_cst.dir_afis_Ukraine
    COUNTRY = 'Ukraine'
    DIR_OUT_FIGS = dm_cst.DIR_OUT_FIGS_NDVI1km_profiles_by_ASAP_admin_by_customAFI_Ukraine
    # xdomain dict
    xdomain = 'gdd' #'time' 'gdd'
    if xdomain == 'time':
        # Time
        DICT_XDOMAIN = {'x': 'time', 'START_YYYYMMDD': 20190101, 'END_YYYYMMDD': 20220321, 'NDVI_FN_TEMPLATE': dm_cst.NDVI_FN_TEMPLATE_time}
        # time domain for the time profile
        # START_YYYYMMDD = 20190101  ## 20150101
        # END_YYYYMMDD = 20220321  # 20220311  ## 20151221
    elif xdomain == 'gdd':
        # GDD
        # offsetYearOfInterest refer to the constant to add to the year in the file name (if cumulation start in oct but the growing year of interest is the following one)
        DICT_XDOMAIN = {'x': 'gdd', 'START_GDD_CUMULATION_MMDD': 1001, 'START_YYYY': 2019, 'offsetYearOfInterest': 1,'NDVI_FN_TEMPLATE': dm_cst.NDVI_FN_TEMPLATE_gdd}
        DICT_GDD_GRID = {'MIN': 0, 'MAX': 500, 'N_STEP': 51}
    test = 1

    pos_histo_afi = 'upper right'

# SET USER PARAMETERS HERE
# minimum afi value to be considered for extraction
AFI_THRESHOLD = 0.75#0 #0.75 0.9 #0.5
# With specific it does not compute histos
MASK_TYPE = 'specific' #'specific' #'generic'
CROP_MASK_SCALE = 0.005 # scaling for afi values. From 0-200 to 0-1
pix2km2 = 0.860946376547824
km22ha = 10*10
# number of beans between 0 an 1 for histograms and GAMs
NBINS = 100
# number of max gaussians
NMAXCOMPONENTS = 2
##########################################################################


afis = ['afi_'+ x + '_int.tif' for x in list(DICT_AFI.keys())]
afis_short = list(DICT_AFI.keys())
CROP_MASK_F = [os.path.join(dir_afis, x) for x in afis]
CROP_TYPE_NAME = list(DICT_AFI.keys())


def main():
    if len(CROP_MASK_F) != len(CROP_TYPE_NAME):
        print('Lenght of finame masks and crop types is not equal')
        exit()
    dir_out = os.path.join(DIR_OUT_FIGS, 'Admin_profiles') #'NBINS'+str(NBINS)+'MAXCOMP'+str(NMAXCOMPONENTS))

    CHECK_FOLDER = os.path.isdir(dir_out)
    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(dir_out)
        print("created folder : ", dir_out)

    features = gpd.read_file(ADMIN_SHAPE)
    features = features[features['adm0_name'] == COUNTRY]
    if REGIONS[0] != 'all':
        features = features[features['adm1_name'].isin(REGIONS)]
    features['afi'] = ''
    # define dekads list
    if DICT_XDOMAIN['x'] == 'time':
        dekads_raw = range(tbx_util.date.string2raw(DICT_XDOMAIN['START_YYYYMMDD']), tbx_util.date.string2raw(DICT_XDOMAIN['END_YYYYMMDD']) + 1)
        dekads = [tbx_util.date.raw2dekade(dekade) for dekade in dekads_raw]
        #year = tbx_util.date.dekade2date(dekads[-1]).year
    elif DICT_XDOMAIN['x'] == 'gdd':
        # form the variable part of file name og NDVIxGDD files
        fn_list = glob.glob(dm_cst.NDVI_FN_TEMPLATE_gdd.format('*'))
        dekads = [re.search('xGDD_(.+?).tif', x).group(1) for x in fn_list]
        # gdd_fixed_grid = np.linspace(DICT_GDD_GRID['MIN'], DICT_GDD_GRID['MAX'], num=DICT_GDD_GRID['N_STEP'])
        # fn_out_list = []
        # for g in gdd_fixed_grid.astype('int'):
        #     fn_out_list.append(os.path.normpath(template).format(str(f"{g:03d}")))

    clmns_afi = ['asap1_id', 'adm0_name', 'adm1_name', 'Afi', 'Area_from_afi']
    afi_df = pd.DataFrame(columns=clmns_afi)
    for i, cm in enumerate(CROP_MASK_F):
        print(cm)
        cm_short = afis_short[i]
        profile_df = pd.DataFrame(columns=['asap1_id','adm0_name', 'adm1_name', 'Date', 'Area_from_afi','Avg_NDVI', 'SD_NVDI','hist','hist_bins'])
        # fetch geom mask arrays
        ds_crop_mask = rasterio.open(cm)
        for index, feat in features.iterrows():
            print(feat['adm1_name'])
            crop_mask = tbx_raster.read_masked(ds_crop_mask, mask=feat['geometry'])
            crop_mask = crop_mask * CROP_MASK_SCALE
            features.at[index, 'area_from_afi'] = np.ma.sum(crop_mask) * pix2km2
            if AFI_THRESHOLD > 0:
                crop_mask[crop_mask < AFI_THRESHOLD] = 0
            features.at[index, 'afi'] = crop_mask
            afi_df = pd.concat([afi_df, pd.DataFrame([[feat['asap1_id'], feat['adm0_name'], feat['adm1_name'], cm_short, features.at[index, 'area_from_afi']]], columns=clmns_afi)])
        for dkd in dekads:
            if DICT_XDOMAIN['x'] == 'time':
                print(tbx_util.date.dekade2date(dkd))
                f_path = os.path.normpath(DICT_XDOMAIN['NDVI_FN_TEMPLATE']).format(
                    tbx_util.date.dekade2string(dkd))
                file_values = 'DN'
            elif DICT_XDOMAIN['x'] == 'gdd':
                print(dkd)
                # if dkd == '20181001_500':
                #     print()
                f_path = os.path.normpath(DICT_XDOMAIN['NDVI_FN_TEMPLATE']).format(dkd)
                file_values = 'physical'
            avg, std = extract_ndvi_avg_sd(f_path, features, file_values)
            hists,bins = extract_ndvi_histo(f_path, features, NBINS, file_values)
            tmp = features[['adm0_name','adm1_name']].copy()
            if DICT_XDOMAIN['x'] == 'time':
                tmp['Date'] = tbx_util.date.dekade2date(dkd)
            elif DICT_XDOMAIN['x'] == 'gdd':
                tmp['Date'] = dkd
            tmp['Avg_NDVI'] = avg
            tmp['SD_NVDI']= std
            tmp['hist'] = hists
            tmp['hist_bins'] = bins
            tmp['Area_from_afi'] = features['area_from_afi']
            tmp['asap1_id'] = features['asap1_id']
            profile_df = pd.concat([profile_df, tmp])
            #profile_df = profile_df.append(tmp)
        # get official stats
        # df_OfficialStats = pd.read_csv(fullPathStats)
        # df_wheat = df_OfficialStats[df_OfficialStats['Crop_ID']==2]
        # df_maize = df_OfficialStats[df_OfficialStats['Crop_ID'] == 2]
        # plot and save
        aus = profile_df['adm1_name'].unique()
        colmns= ['Date', 'asap1_id', 'adm0_name', 'adm1_name', 'bins', 'responsibilities']
        dfClassResp = pd.DataFrame(columns=colmns)
        if DICT_XDOMAIN['x'] == 'time':
            profile_df['year'] = pd.DatetimeIndex(profile_df['Date']).year
            # add a field where date has the same year for all (e.g. 2020)
            profile_df['DateSameYear'] = profile_df['Date'].apply(lambda x: x.replace(year=2020))
        elif DICT_XDOMAIN['x'] == 'gdd':
            dekads = [re.search('xGDD_(.+?).tif', x).group(1) for x in fn_list]
            profile_df['year'] = profile_df['Date'].apply(lambda x: x[0:4])
            profile_df['DateSameYear'] = profile_df['Date'].apply(lambda x: x[-3:])
        single_years = profile_df['year'].unique()
        for au in aus:
            #profiles
            # Now I have the dek https://stackoverflow.com/questions/37596714/compare-multiple-year-data-on-a-single-plot-python
            # but perhaps go for option 2 of stack..
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111)
            for yr in single_years:
                profile_df_yr = profile_df[profile_df['year']==yr]
                x = profile_df_yr[profile_df_yr['adm1_name']==au]['DateSameYear'].values
                yrseries = yr
                if DICT_XDOMAIN['x'] == 'gdd':
                    x = x.astype('float')
                    yrseries = int(yr) + DICT_XDOMAIN['offsetYearOfInterest']
                y = profile_df_yr[profile_df_yr['adm1_name']==au]['Avg_NDVI'].values
                e = profile_df_yr[profile_df_yr['adm1_name']==au]['SD_NVDI'].values
                plt.errorbar(x, y, e, linestyle='None', fmt='o', label=str(yrseries), elinewidth=0.2)
            plt.title(au + ', '+ CROP_TYPE_NAME[i] + ', AreaAfi = ' + str('{:,}'.format(int(profile_df[profile_df['adm1_name']==au]['Area_from_afi'].values[0]*km22ha)))+' ha')
            plt.ylabel('NDVI')
            plt.xlabel('Time')
            plt.ylim(0.0,0.9)
            plt.legend(loc='lower right')
            if DICT_XDOMAIN['x'] == 'time':
                plt.xlabel('Time')
                # Define the date format
                date_form = DateFormatter("%m-%d")
                ax.xaxis.set_major_formatter(date_form)
            elif DICT_XDOMAIN['x'] == 'gdd':
                plt.xlabel('GDD form MM' + str(DICT_XDOMAIN['START_GDD_CUMULATION_MMDD'])[0:2] + ' - DD' + str(DICT_XDOMAIN['START_GDD_CUMULATION_MMDD'])[2:] + ' YYYY-1')
                plt.xticks(np.linspace(0, max(x) , 11))
            #plot a small afi histo
            afi_au = features[features['adm1_name'] == au]['afi'].values[0] *100
            fn = os.path.join(dir_out, au + '_' + CROP_TYPE_NAME[i] + '_NDVI_profile_' + DICT_XDOMAIN['x'] + '.png')
            if AFI_THRESHOLD > 0:
                fn = os.path.join(dir_out, au + '_' + CROP_TYPE_NAME[i] + '_NDVI_profile_' + 'afi_threshold_' +  str(AFI_THRESHOLD) +'_' + DICT_XDOMAIN['x'] + '.png')
            if np.ma.sum(afi_au) > 0:
                inax = inset_axes(ax, width="30%",  height=1., loc=pos_histo_afi)# height : 1 inch
                inax.yaxis.tick_right()
                inax.yaxis.set_label_position("right")
                nbins = 50
                bins = np.linspace(0, 100, nbins, endpoint=True)
                # exclude 0
                bins[0] = 0.0000001
                afi_au = afi_au.compressed()
                afi_au = afi_au[afi_au != 0]
                plt.hist(afi_au, bins=bins, density=False)  # , histtype='stepfilled', alpha=0.4)
                if AFI_THRESHOLD > 0:
                    ymin, ymax = plt.ylim()
                    plt.plot([AFI_THRESHOLD*100,AFI_THRESHOLD*100],[ymin, ymax ], 'r-')
                plt.xlabel('AFI>0')
                plt.ylabel('Counts')
            #fn = os.path.join(dir_out, au + '_' + CROP_TYPE_NAME[i] + '_NDVI_profile_'+str(year)+'.png')
            #plt.tight_layout()
            fig.savefig(fn)
            plt.close()
            #histograms are by dek
            au_df = profile_df[profile_df['adm1_name']==au]
            for index, row in au_df.iterrows():
                date = row['Date']
                pltBin = row['hist_bins']
                pltHist = row['hist']
                # plt.bar(pltBin[:-1], pltHist, width=pltBin[0] - pltBin[1], align='edge', edgecolor='black')#, log=True)
                # #plt.plot(pltBin[:-1], pltHist)#, width=1)
                # plt.title(au + ',' +str(date))
                # plt.ylabel('Count')
                # plt.xlabel('NDVI')
                # fn = os.path.join(dir_out, au + '_' + CROP_TYPE_NAME[i] + '_NDVI_zhisto_'+str(date)+'.png')
                # plt.savefig(fn)
                # plt.close()
                # GMM analysis
                if MASK_TYPE == 'generic':
                    fig, responsibilities = gaussian_mixture.gam(pltHist, NMAXCOMPONENTS, au, date, bins=pltBin)
                    fn = os.path.join(dir_out, au + '_' + CROP_TYPE_NAME[i] + '_NDVI_zgam'+str(NMAXCOMPONENTS)+'_' + str(date) + '.png')
                    fig.savefig(fn)
                    plt.close(fig)
                    #plt.close(fig)
                    # now save a df to be used for mapping the classes on NDVI image
                    list2append = [[date, row['asap1_id'], row['adm0_name'], \
                                                       row['adm1_name'], pltBin, responsibilities]]
                    #dfClassResp = dfClassResp.append(pd.DataFrame(list2append, columns=colmns), ignore_index=True)
                    dfClassResp = pd.concat([dfClassResp, pd.DataFrame(list2append, columns=colmns)], ignore_index=True)


    afi_df = afi_df.sort_values(by=['adm1_name','Afi'])
    fn = os.path.join(dir_out, 'crop_type_fraction.csv')
    afi_df.to_csv(fn)

    if MASK_TYPE == 'generic':
        fn = os.path.join(dir_out, 'responsibilities.pkl')
        dfClassResp.to_pickle(fn)
    print('End')

def extract_ndvi_avg_sd(f_path, features, file_values):
    if file_values == 'DN':
        ds_scale = 0.0048
        ds_offset = -0.2000
    elif file_values =='physical':
        ds_scale = 1.0
        ds_offset = 0
      # # fetch dekad file
    # f_path = os.path.normpath('//ies/d5/asap/asap.5.0/data/indicators_ndvi/ndvi/ndvi_{}.img').format(
    #     tbx_util.date.dekade2string(dekad))
    ds = rasterio.open(f_path)
    avg = []
    std = []
    for i, feat in features.iterrows():
       # print(feat['adm1_name'])
        # read datasets with geometry mask
        data = tbx_raster.read_masked(ds, mask=feat['geometry'])
        # create the dataset nodata mask
        if file_values == 'DN':
            no_data_mask = data.data > 250
        elif file_values == 'physical':
            no_data_mask = np.isnan(data.data)
        # include the nodata mask in the dataset mask
        data.mask = np.logical_or(data.mask, no_data_mask)
        # convert the data to native values
        ds_data = (data * ds_scale) + ds_offset
        if np. logical_and(np.sum(feat['afi']) != 0, len(ds_data[ds_data.mask == False])>0):
            wavg = np.ma.average(ds_data,  weights=feat['afi'])
            avg.append(wavg)
            std.append(np.ma.sqrt(np.ma.average((ds_data - wavg) ** 2, weights=feat['afi'])))
        else:
            avg.append(np.NaN)
            std.append(np.NaN)
    return avg, std

def extract_ndvi_histo(f_path, features, nbins, file_values):
    if file_values == 'DN':
        ds_scale = 0.0048
        ds_offset = -0.2000
    elif file_values == 'physical':
        ds_scale = 1.0
        ds_offset = 0
    #bins
    binValues = np.linspace(0, 1, nbins, endpoint=True)
    # fetch dekad file
    #f_path = os.path.normpath('//ies/d5/asap/asap.5.0/data/indicators_ndvi/ndvi/ndvi_{}.img').format(
        #tbx_util.date.dekade2string(dekad))
    ds = rasterio.open(f_path)
    hists = []
    bins = []
    for i, feat in features.iterrows():
        # read datasets with geometry mask
        data = tbx_raster.read_masked(ds, mask=feat['geometry'])
        # create the dataset nodata mask
        if file_values == 'DN':
            no_data_mask = data.data > 250
        elif file_values == 'physical':
            no_data_mask = np.isnan(data.data)
        # include the nodata mask in the dataset mask
        data.mask = np.logical_or(data.mask, no_data_mask)
        # convert the data to native values
        ds_data = (data * ds_scale) + ds_offset
        # make histogram
        # Extract data and weights on locations where both are not masked
        mask_of_d = np.ma.getmask(ds_data)
        mask_of_w = np.ma.getmask(feat['afi'])
        mask_intersect = np.logical_or(mask_of_d, mask_of_w)
        hist, bin = np.histogram(ds_data[~mask_intersect], bins = binValues, weights=feat['afi'][~mask_intersect])
        hists.append(hist)
        bins.append(bin)
    return hists, bins

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    desired_width = 700
    pd.set_option('display.width', desired_width)
    start = dt.datetime.now()
    main()
    print('Execution time:', dt.datetime.now() - start)