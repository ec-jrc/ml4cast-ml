import pandas as pd


def retain_X(x, statsX, c):
    regionID_list = statsX[statsX['Crop_ID'] == c]['Region_ID']
    xc = x[(x['Crop_ID']==c) & (x['AU_code'].isin(regionID_list))]
    #print('function retain_90: '+ 'Crop ' + str(xc['Crop_ID'].iloc[0]) + ', ' + xc['Crop_name'].iloc[0])
    #print('Regions', *xc['AU_name'].unique(), sep=", ")
    #print('---------------')
    return xc