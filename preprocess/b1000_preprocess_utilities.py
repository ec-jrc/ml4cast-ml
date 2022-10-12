def retain_X(x, statsX, c):
    # get data from a certain crop and certain regions
    regionID_list = statsX[statsX['Crop_ID'] == c]['Region_ID']
    xc = x[(x['Crop_ID']==c) & (x['AU_code'].isin(regionID_list))]
    return xc