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
from dynamic_masking import gaussian_mixture
import dm_local_constants as dm_cst

##########################################################################
# SET INPUTS HERE

study = 'Ukraine'

if study == 'Ukraine':
    ADMIN_SHAPE = dm_cst.ADMIN_SHAPE_Ukraine_gaul1   #r'X:\PY_data\Ukraine\admin_bound\Ukraine_gaul1.shp'
    # dictionary can be taken from AFI_maker.py, selct the afis you want
    # DICT_AFI = {'wheat_2019': [2],
    #             'maize_2019': [5]}
    #DICT_AFI = {'arable_2019': [2, 3, 4, 5, 6, 7, 8, 9, 15, 16]}
    DICT_AFI = {'winter_2019': [2, 3, 9, 15],
                'spring_2019': [4],
                'summer_2019': [5, 6, 7, 8, 16]}
    DICT_AFI = {'winter_2016': [2, 3, 9, 15],
                'spring_2016': [4],
                'summer_2016': [5, 6, 7, 8, 16]}
    REGIONS = ["Khersons'ka"] #['all'] #["Cherkas'ka"] #['all']
    dir_afis = dm_cst.dir_afis_Ukraine
    COUNTRY = 'Ukraine'
    DIR_OUT_FIGS = dm_cst.DIR_OUT_FIGS_Ukraine
    # time domain for the gmm
    START_YYYYMMDD = 20160301  ## 20150101
    END_YYYYMMDD = 20160301 #20220311  ## 20151221
    pos_histo_afi = 'upper right'
    AIC_THRESHOLD = -200 #temporay to esclude a higher num of component when AIC diff is not big

# SET USER PARAMETERS HERE
# minimum afi value to be considered for extraction
AFI_THRESHOLD = 0.5#0 #0.75 0.9 #0.5
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
        print('Lenght of filename masks and crop types is not equal')
        exit()
    dir_out = os.path.join(DIR_OUT_FIGS, 'NBINS'+str(NBINS)+'MAXCOMP'+str(NMAXCOMPONENTS))

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
    dekads_raw = range(tbx_util.date.string2raw(START_YYYYMMDD), tbx_util.date.string2raw(END_YYYYMMDD) + 1)
    dekads = [tbx_util.date.raw2dekade(dekade) for dekade in dekads_raw]
    year = tbx_util.date.dekade2date(dekads[-1]).year
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
            print(tbx_util.date.dekade2date(dkd))
            avg, std = extract_ndvi_avg_sd(dkd, features)
            hists,bins = extract_ndvi_histo(dkd, features, NBINS)
            tmp = features[['adm0_name','adm1_name']].copy()
            tmp['Date'] = tbx_util.date.dekade2date(dkd)
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
        for au in aus:
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
                if np.count_nonzero(pltHist) < 2:
                    print(au + 'has 0 or only one bin' + CROP_TYPE_NAME[i])
                else:
                    fig, responsibilities = gaussian_mixture.gam(pltHist, NMAXCOMPONENTS, au, date, bins=pltBin, aic_threshold = AIC_THRESHOLD)
                    fn = os.path.join(dir_out, au + '_' + CROP_TYPE_NAME[i] + '_NDVI_zgam'+str(NMAXCOMPONENTS)+'_' + str(date) + '.png')
                    fig.savefig(fn)
                    plt.close(fig)
                #plt.close(fig)
                # now save a df to be used for mapping the classes on NDVI image
                list2append = [[date, row['asap1_id'], row['adm0_name'], \
                                                   row['adm1_name'], pltBin, responsibilities]]
                #dfClassResp = dfClassResp.append(pd.DataFrame(list2append, columns=colmns), ignore_index=True)
                dfClassResp = pd.concat([dfClassResp, pd.DataFrame(list2append, columns=colmns)], ignore_index=True)



        fn = os.path.join(dir_out, 'responsibilities_' + CROP_TYPE_NAME[i] + '_NDVI_zgam'+str(NMAXCOMPONENTS)+'_' + str(date) +'.pkl')
        dfClassResp.to_pickle(fn)
        #fn = os.path.join(dir_out, 'responsibilities' + CROP_TYPE_NAME[i] + '_NDVI_zgam' + str(NMAXCOMPONENTS) + '_' + str(date) + '.csv')
        #dfClassResp.to_csv(fn)
    print('End')

def extract_ndvi_avg_sd(dekad, features):
    # src def
    ds_scale = 0.0048
    ds_offset = -0.2000
    # fetch dekad file
    f_path = os.path.normpath('//ies/d5/asap/asap.5.0/data/indicators_ndvi/ndvi/ndvi_{}.img').format(
        tbx_util.date.dekade2string(dekad))
    ds = rasterio.open(f_path)
    avg = []
    std = []
    for i, feat in features.iterrows():
       # print(feat['adm1_name'])
        # read datasets with geometry mask
        data = tbx_raster.read_masked(ds, mask=feat['geometry'])
        # create the dataset nodata mask
        no_data_mask = data.data > 250
        # include the nodata mask in the dataset mask
        data.mask = np.logical_or(data.mask, no_data_mask)
        # convert the data to native values
        ds_data = (data * ds_scale) + ds_offset
        if np.sum(feat['afi']) != 0:
            wavg = np.ma.average(ds_data,  weights=feat['afi'])
            avg.append(wavg)
            std.append(np.ma.sqrt(np.ma.average((ds_data - wavg) ** 2, weights=feat['afi'])))
        else:
            avg.append(np.NaN)
            std.append(np.NaN)
    return avg, std

def extract_ndvi_histo(dekad, features, nbins):
    # src def
    ds_scale = 0.0048
    ds_offset = -0.2000
    #bins
    binValues = np.linspace(0, 1, nbins, endpoint=True)
    # fetch dekad file
    f_path = os.path.normpath('//ies/d5/asap/asap.5.0/data/indicators_ndvi/ndvi/ndvi_{}.img').format(
        tbx_util.date.dekade2string(dekad))
    ds = rasterio.open(f_path)
    hists = []
    bins = []
    for i, feat in features.iterrows():
        # read datasets with geometry mask
        data = tbx_raster.read_masked(ds, mask=feat['geometry'])
        # create the dataset nodata mask
        no_data_mask = data.data > 250
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