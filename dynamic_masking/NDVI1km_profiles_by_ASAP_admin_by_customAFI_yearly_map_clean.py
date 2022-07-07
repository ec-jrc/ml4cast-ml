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
from matplotlib.ticker import MultipleLocator
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
'ZAeast' South Africa east map (4 provinces)
'''

study = 'Ukraine' # 'ZAeast' 

##########################################################################
# SET INPUTS HERE

if study == 'Ukraine':
    # GEO DOMAIN
    ADMIN_SHAPE = dm_cst.ADMIN_SHAPE_Ukraine_gaul1
    REGIONS = ['all']  #["Khersons'ka", "Kharkivs'ka"] #['all']  # ["Khersons'ka"] # ['all'] #["Khersons'ka", "Kharkivs'ka"]
    COUNTRY = 'Ukraine'

    # I/O DETAILS
    dir_afis = dm_cst.dir_afis_Ukraine
    DIR_OUT_FIGS = dm_cst.DIR_OUT_FIGS_NDVI1km_profiles_by_ASAP_admin_by_customAFI_yearly_map_clean

    # X DOMAIN
    xdomain = 'time'  # 'time' 'gdd'
    if xdomain == 'time':
        # Time (now it can only start in 0101 (be care if changing to approach used to link to yeraly mask (the year from a dkd is used to link)
        DICT_XDOMAIN = {'x': 'time', 'START_YYYYMMDD': 20190101, 'END_YYYYMMDD': 20220411,
                        'NDVI_FN_TEMPLATE': dm_cst.NDVI_FN_TEMPLATE_time}
    elif xdomain == 'gdd':
        # offsetYearOfInterest refer to the constant to add to the year in the file name (if cumulation start in oct but the growing year of interest is the following one)
        DICT_XDOMAIN = {'x': 'gdd', 'START_GDD_CUMULATION_MMDD': 1001, 'START_YYYY': 2019, 'offsetYearOfInterest': 1,
                        'NDVI_FN_TEMPLATE': dm_cst.NDVI_FN_TEMPLATE_gdd}
        DICT_GDD_GRID = {'MIN': 0, 'MAX': 5000, 'N_STEP': 251} # DICT_GDD_GRID = {'MIN': 0, 'MAX': 500, 'N_STEP': 51}

    # AFIs to USE
    # The code work with yearly crop mask. If you only have one (static mask) it will be repetated every year.
    # DICT_AFI = {'2019': ['winter', [2, 3]],
    #             '2020': ['winter', [2, 3]],
    #             '2021': ['winter', [2, 3]],
    #             '2022': ['winter', [2, 3]]}
    DICT_AFI = {'2019': ['arable', [2,3,4,5,6,7,8,9,15,16,17]]}
    if len(list(DICT_AFI.keys())) > 1:
        ISYEARLYAFI = True
    else:
        ISYEARLYAFI = False

if study == 'ZAeast':
    # GEO DOMAIN
    ADMIN_SHAPE = dm_cst.ADMIN_SHAPE_Global_gaul1
    REGIONS = ['North West', 'Gauteng', 'Mpumalanga', 'Free State']
    COUNTRY = 'South Africa'

    # I/O DETAILS
    dir_afis = dm.cst.dir_afis_ZAeast
    DIR_OUT_FIGS = dm.cst.dir_afis_ZAeast

    # X DOMAIN
    xdomain = 'time'  # 'time' 'gdd'
    if xdomain == 'time':
        # Time (now it can only start in 0101 (be care if changing to approach used to link to yeraly mask (the year from a dkd is used to link)
        DICT_XDOMAIN = {'x': 'time', 'START_YYYYMMDD': 20180101, 'END_YYYYMMDD': 20220401,
                        'NDVI_FN_TEMPLATE': dm_cst.NDVI_FN_TEMPLATE_time}
    # elif xdomain == 'gdd':
    #     # offsetYearOfInterest refer to the constant to add to the year in the file name (if cumulation start in oct but the growing year of interest is the following one)
    #     DICT_XDOMAIN = {'x': 'gdd', 'START_GDD_CUMULATION_MMDD': 1001, 'START_YYYY': 2019, 'offsetYearOfInterest': 1,
    #                     'NDVI_FN_TEMPLATE': dm_cst.NDVI_FN_TEMPLATE_gdd}
    #     DICT_GDD_GRID = {'MIN': 0, 'MAX': 5000, 'N_STEP': 251} # DICT_GDD_GRID = {'MIN': 0, 'MAX': 500, 'N_STEP': 51}

    # AFIs to USE
    # The code work with yearly crop mask. If you only have one (static mask) it will be repetated every year.

    DICT_AFI = {'2019': ['arable', [1]]}
    if len(list(DICT_AFI.keys())) > 1:
        ISYEARLYAFI = True
    else:
        ISYEARLYAFI = False

# GENERAL SETTINGS
# where to place insert
pos_histo_afi = 'upper right'
# minimum afi value to be considered for extraction
AFI_THRESHOLD = 0.75 #0 #0.75 0.9 #0.5
# With specific it does not compute histos
CROP_MASK_SCALE = 0.005 # scaling for afi values. From 0-200 to 0-1
pix2km2 = 0.860946376547824
km22ha = 10*10
# number of beans between 0 an 1 for histograms and GAMs
NBINS = 100
# number of max gaussians
NMAXCOMPONENTS = 2
##########################################################################
CROP_MASK_F = ['']
CROP_TYPE_NAME = list(DICT_AFI.items())[0][1][0]

def main():
    dir_out = os.path.join(DIR_OUT_FIGS, 'Admin_profiles') #'NBINS'+str(NBINS)+'MAXCOMP'+str(NMAXCOMPONENTS))
    CHECK_FOLDER = os.path.isdir(dir_out)
    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(dir_out)
        print("created folder: ", dir_out)
    # Read geometries
    features = gpd.read_file(ADMIN_SHAPE)
    features = features[features['adm0_name'] == COUNTRY]
    if REGIONS[0] != 'all':
        features = features[features['adm1_name'].isin(REGIONS)]
    # define dekads list (in case gdd selected is not dekad anymore)
    if DICT_XDOMAIN['x'] == 'time':
        dekads_raw = range(tbx_util.date.string2raw(DICT_XDOMAIN['START_YYYYMMDD']), tbx_util.date.string2raw(DICT_XDOMAIN['END_YYYYMMDD']) + 1)
        dekads = [tbx_util.date.raw2dekade(dekade) for dekade in dekads_raw]
    elif DICT_XDOMAIN['x'] == 'gdd':
        # form the variable part of file name of NDVIxGDD files
        fn_list = glob.glob(dm_cst.NDVI_FN_TEMPLATE_gdd.format('*'))
        dekads = [re.search('xGDD_(.+?).tif', x).group(1) for x in fn_list]
    # understand how many year of AFIs are required
    yearOfAFI = sorted(list(set([int(str(x)[0:4]) for x in dekads])))
    if DICT_XDOMAIN['x'] == 'gdd':
        yearOfAFI = [x + DICT_XDOMAIN['offsetYearOfInterest'] for x in yearOfAFI]
    yearOfAFI = [str(x) for x in yearOfAFI]
    # Define AFIs
    #yearOfAFI = list(DICT_AFI.keys())
    features[['afi' + x for x in yearOfAFI]] = ''
    clmns_afi = ['asap1_id', 'adm0_name', 'adm1_name', 'Afi']
    clmns_afi.extend(['Area_from_afi' + x for x in yearOfAFI])
    afi_df = pd.DataFrame(columns=clmns_afi)
    profile_df = pd.DataFrame(columns=['asap1_id','adm0_name', 'adm1_name', 'Date', 'Area_from_afi','Avg_NDVI', 'SD_NVDI','nValid', 'hist','hist_bins','lastGDDof05Area'])
    # Read various AFIs
    for yr in yearOfAFI:
        # yeraly crop masks, store afis for all
        if ISYEARLYAFI == True:
            tmp = 'afi_' + CROP_TYPE_NAME + '_' + yr + '_int.tif'
        else:
            tmp = 'afi_' + CROP_TYPE_NAME + '_' + list(DICT_AFI.keys())[0] + '_int.tif'
        # afis_short = list(DICT_AFI.keys())
        fn_cm = os.path.join(dir_afis, tmp)
        ds_crop_mask = rasterio.open(fn_cm)
        for index, feat in features.iterrows():
            print(feat['adm1_name'])
            crop_mask = tbx_raster.read_masked(ds_crop_mask, mask=feat['geometry'])
            crop_mask = crop_mask * CROP_MASK_SCALE
            features.at[index, 'area_from_afi'+yr] = np.ma.sum(crop_mask) * pix2km2
            if AFI_THRESHOLD > 0:
                crop_mask[crop_mask < AFI_THRESHOLD] = 0
            features.at[index, 'afi'+yr] = crop_mask
            afi_df = pd.concat([afi_df, pd.DataFrame([[feat['asap1_id'], feat['adm0_name'], feat['adm1_name'],
                                                       CROP_TYPE_NAME + yr, features.at[index, 'area_from_afi'+yr]]],
                                                     columns=['asap1_id', 'adm0_name', 'adm1_name', 'Afi', 'Area_from_afi' + yr])])
    # Read NDVI profiles
    for dkd in dekads:
        # here I have to associate the correct afi to the dek
        if DICT_XDOMAIN['x'] == 'time':
            print(tbx_util.date.dekade2date(dkd))
            f_path = os.path.normpath(DICT_XDOMAIN['NDVI_FN_TEMPLATE']).format(
                tbx_util.date.dekade2string(dkd))
            file_values = 'DN'
            year2use = str(tbx_util.date.dekade2date(dkd).year)
        elif DICT_XDOMAIN['x'] == 'gdd':
            print(dkd)
            f_path = os.path.normpath(DICT_XDOMAIN['NDVI_FN_TEMPLATE']).format(dkd)
            file_values = 'physical'
            year2use = str(int(dkd[0:4]) + DICT_XDOMAIN['offsetYearOfInterest'])

        features['afi'] = features['afi' + year2use]
        features['area_from_afi'] = features['area_from_afi' + year2use]
        avg, std, nValid = extract_ndvi_avg_sd(f_path, features, file_values)
        # hists,bins = extract_ndvi_histo(f_path, features, NBINS, file_values)
        tmp = features[['adm0_name','adm1_name']].copy()
        if DICT_XDOMAIN['x'] == 'time':
            tmp['Date'] = tbx_util.date.dekade2date(dkd)
        elif DICT_XDOMAIN['x'] == 'gdd':
            tmp['Date'] = dkd
        tmp['Avg_NDVI'], tmp['SD_NVDI'],  tmp['nValid'], tmp['Area_from_afi'], tmp['asap1_id'] = avg, std, nValid, features['area_from_afi'], features['asap1_id']
        profile_df = pd.concat([profile_df, tmp])

    # plot and save
    if DICT_XDOMAIN['x'] == 'time':
        profile_df['year'] = pd.DatetimeIndex(profile_df['Date']).year
        # add a field where date has the same year for all (e.g. 2020)
        profile_df['DateSameYear'] = profile_df['Date'].apply(lambda x: x.replace(year=2020))
    elif DICT_XDOMAIN['x'] == 'gdd':
        profile_df['year'] = profile_df['Date'].apply(lambda x: x[0:4])
        profile_df['DateSameYear'] = profile_df['Date'].apply(lambda x: x.split('_')[1])
    single_years = profile_df['year'].unique()
    for au in profile_df['adm1_name'].unique():
        print(au)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        for yr in single_years:
            profile_df_yr = profile_df[profile_df['year']==yr]
            x = profile_df_yr[profile_df_yr['adm1_name']==au]['DateSameYear'].values
            #strAreaFromAfi = str('{:,}'.format(int(profile_df[profile_df['adm1_name'] == au]['Area_from_afi'].values[0] * km22ha/1000))) + ' kha'
            yrseries = yr
            if DICT_XDOMAIN['x'] == 'gdd':
                x = x.astype('float')
                yrseries = int(yr) + DICT_XDOMAIN['offsetYearOfInterest']
            strAreaFromAfi = str('{:,}'.format(int(profile_df_yr[profile_df_yr['adm1_name'] == au]['Area_from_afi'].values[0] * km22ha/1000))) + ' kha'
            yrseries = str(yrseries) + ' ' + strAreaFromAfi
            y = profile_df_yr[profile_df_yr['adm1_name']==au]['Avg_NDVI'].values
            e = profile_df_yr[profile_df_yr['adm1_name']==au]['SD_NVDI'].values
            # plot differently if it is the last year and gdd was requested
            if yr == single_years[-1] and DICT_XDOMAIN['x'] == 'gdd':
                nV = profile_df_yr[profile_df_yr['adm1_name'] == au]['nValid'].values * pix2km2 * km22ha/1000.0
                nv_scaled = nV/nV[0]
                plt.errorbar(x[0], y[0], e[0], linestyle='None', fmt='o', label=str(yrseries), elinewidth=0.2, color='red')#, markersize=mks)
                plt.errorbar(x[1:], y[1:], e[1:], linestyle='None', fmt='o', elinewidth=0.2, color='red', markerfacecolor='white', zorder=2)#, markersize=mks)
                plt.scatter(x[1:], y[1:], alpha=nv_scaled[1:], color='red', zorder=10)#, s=mks)
            else:
                plt.errorbar(x, y, e, linestyle='None', fmt='o', label=str(yrseries), elinewidth=0.2, zorder=3)#, markersize=mks
        plt.title(au + ', Crop group: ' + CROP_TYPE_NAME)
        plt.ylabel('NDVI')
        plt.ylim(0.0,0.9)
        plt.legend(loc='lower right')
        if DICT_XDOMAIN['x'] == 'time':
            plt.xlabel('Time')
            # Define the date format
            date_form = DateFormatter("%m-%d")
            ax.xaxis.set_major_formatter(date_form)
        elif DICT_XDOMAIN['x'] == 'gdd':
            plt.xlim(0.0, 5000)
            plt.xlabel('GDD form MM' + str(DICT_XDOMAIN['START_GDD_CUMULATION_MMDD'])[0:2] + ' - DD' + str(DICT_XDOMAIN['START_GDD_CUMULATION_MMDD'])[2:] + ' YYYY-1')
            plt.xticks(np.linspace(0, max(x), 11))
            ax.xaxis.set_minor_locator(MultipleLocator(100))

        #plot a small afi histo
        name = au + '_' + CROP_TYPE_NAME + '_NDVI_' + DICT_XDOMAIN['x'] + '_profile'
        if AFI_THRESHOLD > 0:
            name = name + '_afi_threshold_' + str(AFI_THRESHOLD)
        if ISYEARLYAFI == True:
            name = name + '_yearly_CM'
        else:
            name = name + '_static_CM'
        fn = os.path.join(dir_out, name + '.png')
        #if yearlyAFI == True and DICT_XDOMAIN['x'] == 'gdd':
        if DICT_XDOMAIN['x'] == 'gdd':
            inax = inset_axes(ax, width="30%", height=1., loc=pos_histo_afi)  # height : 1 inch
            inax.yaxis.tick_left()
            inax.yaxis.set_label_position("left")
            plt.tick_params(axis='x', labelsize=6)
            plt.tick_params(axis='y', labelsize=6)
            if AFI_THRESHOLD > 0:
                plt.ylabel('Area (kha), AFI>'+str(AFI_THRESHOLD), fontdict={'fontsize': 6})
            else:
                plt.ylabel('Area GDD (kha)', fontdict={'fontsize': 6})
            plt.xlabel('GDD', fontdict={'fontsize': 8})

            for yr in single_years:
                profile_df_yr = profile_df[profile_df['year'] == yr]
                x = profile_df_yr[profile_df_yr['adm1_name'] == au]['DateSameYear'].values
                x = x.astype('float')
                nV = profile_df_yr[profile_df_yr['adm1_name'] == au]['nValid'].values * pix2km2 * km22ha/1000.0
                plt.plot(x,nV,'-o', markersize=2)
                plt.xticks(np.linspace(0, max(x), 51))
            #limit y to last
            plt.xlim(0, x[np.where(nV==0)[0][0]])#-1])
            # and annotate
            lastyy = -999
            for xt, yt in zip(x, nV):
                if yt != lastyy:
                    plt.annotate(np.around(yt, decimals=1), (xt, yt + 10), fontsize=6)
                lastyy = yt
            lastGDDofEqualArea = x[np.where(nV / nV[0] == 1)[0][-1]]
            lastGDDof05Area = x[np.where(nV / nV[0] > 0.5)[0][-1]]
            profile_df.loc[profile_df['adm1_name']==au,['lastGDDof05Area']] = lastGDDof05Area
            ax.plot([lastGDDofEqualArea, lastGDDofEqualArea], [0, 1], linestyle='dotted', color='black', zorder=1)
            ax.annotate('Last GDD with full area available', (lastGDDofEqualArea + 30, 0.01), fontsize=8)
        if ISYEARLYAFI == False and DICT_XDOMAIN['x'] == 'time':
            afi_au = features[features['adm1_name'] == au]['afi'].values[0] *100
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

        fig.savefig(fn)
        plt.close()


    # shortcut to map the anomalies at last dekad of current year
    # I have to map GDD time lastGDDof05Area
    # save in a df the values of last three data point for all year for all admin units
    colmns = ['asap1_id', 'adm0_name', 'adm1_name', 'Date', 'year', 'DateSameYear', 'percentOfMean']
    df_anomalies = pd.DataFrame(columns=colmns)
    minGDD = 10000000
    maxGDD = -10
    for au in profile_df['adm1_name'].unique():
        print(au)
        profile_df_au = profile_df[profile_df['adm1_name'] == au]
        profile_df_au = profile_df_au.sort_values(by=['Date'], axis=0)
        # get rid of non existing values when using gdd (last year)
        profile_df_au.dropna(subset=['Avg_NDVI'], inplace=True)
        # get year and date (can be real date of the last
        yearLast = profile_df_au['year'].iloc[-1]
        DateSameYearLast = profile_df_au['DateSameYear'].iloc[-1]
        #print(DateSameYearLast)
        if DICT_XDOMAIN['x'] == 'gdd':
            #DateSameYearLast = f"{int(lastGDDof05Area):04d}"
            DateSameYearLast = profile_df_au['lastGDDof05Area'].iloc[0]
            DateSameYearLast = f"{int(DateSameYearLast):04d}"
            minGDD = np.min(np.array([minGDD,DateSameYearLast]).astype('float'))
            maxGDD = np.max(np.array([maxGDD, DateSameYearLast]).astype('float'))
        refYears = [el for el in profile_df_au['year'].unique() if el != yearLast]
        avg = np.mean(profile_df_au[(profile_df_au['year'].isin(refYears)) & (profile_df_au['DateSameYear']==DateSameYearLast)]['Avg_NDVI'].values)
        val = profile_df_au[(profile_df_au['year']==yearLast) & (profile_df_au['DateSameYear']==DateSameYearLast)]['Avg_NDVI'].values[0]
        row = profile_df_au[['asap1_id', 'adm0_name', 'adm1_name', 'Date', 'year', 'DateSameYear']].iloc[-1].tolist() + [val/avg*100]
        df_anomalies = pd.concat([df_anomalies, pd.DataFrame([row], columns=colmns)], ignore_index=True)
    # map it
    map_df = features.merge(df_anomalies, how='left', left_on="asap1_id", right_on="asap1_id")
    if DICT_XDOMAIN['x'] == 'time':
        text = "Reference date = " + str(yearLast) + ' ' + str(DateSameYearLast)
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        ax.set_title(text)
    elif DICT_XDOMAIN['x'] == 'gdd':
        #text = "Reference GDD" + ' (from ' + str(DICT_XDOMAIN['START_GDD_CUMULATION_MMDD']) + ' YYYY-1) = ' + str(int(minGDD)) + '-' + str(int(maxGDD))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))#, sharex=True, sharey=True)


    map_df['zz'] = pd.to_numeric(map_df["percentOfMean"])
    #plt.sca(ax[0])
    #map_df.plot(column='percentOfMean', ax=ax, legend=True, legend_kwds={'label': "test", 'orientation': "horizontal", 'shrink': 0.4}, vmin=50, vmax=150)
    ttmp = [str(int(x)+1) for x in refYears]
    map_df.plot(column='zz', ax=ax1, categorical=False, legend=True, cmap='bwr_r', edgecolor='k',
                legend_kwds={'label': "% of avg NDVI (Years="+ ','.join([str(x) for x in ttmp]) + ")", 'orientation': "horizontal", 'shrink': 0.4, 'extend': 'both'},
                vmin=50, vmax=150)
    ax1.set_ylabel('Deg N')
    ax1.set_xlabel('Deg E')
    # start, end = ax1.get_xlim()
    # ss = np.floor(start)
    # ee = np.ceil(end)
    # ax1.set_xlim(ss, ee)
    # ax1.set_xticks(np.arange(ss, ee, 4))
    #ax1.set_xlim(start, end)

    # start, end = ax1.get_ylim()
    # ss = np.floor(start)
    # ee = np.ceil(end)
    # ax1.set_ylim(ss, ee)
    # ax1.set_yticks(np.arange(ss, ee, 1))
    #ax1.set_ylim(start, end)


    if DICT_XDOMAIN['x'] == 'gdd':
        map_df['kk'] = pd.to_numeric(map_df["DateSameYear"])
        map_df.plot(column='kk', ax=ax2, categorical=False, legend=True, cmap='YlOrBr', edgecolor='k',
                    legend_kwds={'label': "Reference GDD",
                                 'orientation': "horizontal", 'shrink': 0.4, 'extend': 'both'},
                    vmin=int(np.floor(map_df['kk'].min() / 500.0))*500, vmax=int(np.ceil(map_df['kk'].max() / 500.0))*500)
        ax2.set_ylabel('Deg N')
        ax2.set_xlabel('Deg E')
        fig.suptitle = 'NDVI anomalies at reference GDD (larger GDD reached by at least 50% area)'

        # start, end = ax2.get_xlim()
        # ss = np.floor(start)
        # ee = np.ceil(end)
        # ax2.set_xlim(ss, ee)
        # ax2.set_xticks(np.arange(ss, ee, 4))
        # # ax1.set_xlim(start, end)
        #
        # start, end = ax2.get_ylim()
        # ss = np.floor(start)
        # ee = np.ceil(end)
        # #ax2.set_ylim(ss, ee)
        # print(np.arange(ss, ee, 1))
        # ax2.set_yticks(np.arange(ss, ee, 1))
        # ax2.set_yticklabels(np.arange(ss, ee, 1).astype('str'))
        # ax2.yaxis.tick_right()
        # ax2.yaxis.set_label_position("right")
    name = CROP_TYPE_NAME

    if DICT_XDOMAIN['x'] == 'time':
        name = name + '_anomaly_map_time'
    elif DICT_XDOMAIN['x'] == 'gdd':
        name = name + '_anomaly_map_gdd'
    if ISYEARLYAFI == True:
        fn = os.path.join(dir_out, CROP_TYPE_NAME + name + '_yearly_CM.png')
    else:
        fn = os.path.join(dir_out, CROP_TYPE_NAME + name + '_static_CM.png')

    plt.tight_layout()
    fig.savefig(fn)  # , dpi = 300)
    plt.close(fig)

    print()
    afi_df = afi_df.sort_values(by=['adm1_name','Afi'])
    fn = os.path.join(dir_out, 'crop_type_fraction.csv')
    afi_df.to_csv(fn)


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
    nV = []
    for i, feat in features.iterrows():
       # print(feat['adm1_name'])
        # read datasets with geometry mask
        data = tbx_raster.read_masked(ds, mask=feat['geometry'])
        # create the dataset nodata mask
        if file_values == 'DN':
            no_data_mask = data.data > 250
            nValid = 0
        elif file_values == 'physical':
            no_data_mask = np.isnan(data.data)
            oneWhereNaN  = ~np.isnan(data.data) & ~data.mask
            nValid = (oneWhereNaN * feat['afi']).sum()
        # include the nodata mask in the dataset mask
        data.mask = np.logical_or(data.mask, no_data_mask)
        # convert the data to native values
        ds_data = (data * ds_scale) + ds_offset
        if np. logical_and(np.sum(feat['afi']) != 0, len(ds_data[ds_data.mask == False])>0):
            wavg = np.ma.average(ds_data,  weights=feat['afi'])
            avg.append(wavg)
            std.append(np.ma.sqrt(np.ma.average((ds_data - wavg) ** 2, weights=feat['afi'])))
            nV.append(nValid)
        else:
            avg.append(np.NaN)
            std.append(np.NaN)
            nV.append(0)

    return avg, std, nV

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