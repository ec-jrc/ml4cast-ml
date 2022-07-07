import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap


import b05_Init
def cluster_by_yield_level_lat_lon(target):
    '''Cluster by yield level and lat lon'''

    desired_width = 320
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns', 10)

    project = b05_Init.init(target)
    dirOut = project['output_dir']
    dirOutCluster = dirOut + '/AU_clusters/'

    dirAmdUn = project['input_dir'] + '/AdminUnits'
    # load shp
    fp = dirAmdUn + '/' + project['AOI'] + '_AdminUnits.shp'
    map_df = gpd.read_file(fp)

    # open stats and predictors
    stats90 = pd.read_pickle(dirOut + '/' + project['AOI'] + '_stats90.pkl')
    #stats = pd.read_pickle(dirOut + '/' + project['AOI'] + '_stats.pkl')
    #timerange = project['timeRange']
    #stats = stats.drop(stats[stats['Year'] < timerange[0]].index)
    n_cluster_by_crop = [7, 7, 6]
    for c in range(1, 4):
        #regionID_list = stats90[stats90['Crop_ID'] == c]['Region_ID']
        #region_names = stats90[stats90['Crop_ID'] == c]['AU_name']
        crop_name = stats90.loc[stats90['Crop_ID'] == c].iloc[0]['Crop_name']
        crop_name = crop_name.values[0]
        print('Crop ' + str(c) + ', ' + crop_name)
        xc = stats90[stats90['Crop_name', 'first'] == crop_name]
        # remove level
        xc.columns = [f'{i}_{j}' if j != '' else f'{i}' for i, j in xc.columns]
        xc = xc.sort_values('Yield_mean')
        # get centroid coord

        map_df['x'] = map_df.centroid.x
        map_df['y'] = map_df.centroid.y
        #xc0 = xc
        xc = xc.merge(map_df[['AU_code', 'x', 'y']], how='left', left_on="Region_ID", right_on="AU_code")

        labels = xc['AU_name_first'].to_list()
        labels = [elem[:8] for elem in labels]
        xc['labels'] = labels

        # keep only relevant variables
        data = xc[['Yield_mean','x','y']].values
        scaler = MinMaxScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        # data.shape
        # data = data.reshape((-1,1))
        # print(data[0, :])
        plt.figure(figsize=(10, 7))
        plt.title(crop_name)
        #plt.xlabel("Region")
        dend = shc.dendrogram(shc.linkage(data, method='ward'), labels=labels, leaf_rotation=90)
        plt.subplots_adjust(bottom=0.15)
        strFn = dirOutCluster + project['AOI'] + '_' + crop_name + '_90prctYield_dendrogram.png'
        plt.savefig(strFn.replace(" ", ""))
        #plt.show()
        plt.close()
        cluster = AgglomerativeClustering(n_clusters=n_cluster_by_crop[c - 1], \
                                          affinity='euclidean', linkage='ward')
        groups = cluster.fit_predict(data)
        columnName = 'Cluster_ID'#'Ward_' + str(n_cluster_by_crop[c - 1]) + 'groups'
        xc[columnName] = groups
        clrlist = np.array(['beige', 'red', 'green', 'lightblue', 'chartreuse', 'blue', 'grey', 'lavander','pink'])

        xdata = list(range(len(xc['AU_name_first'])))
        plt.bar(xdata, xc['Yield_mean'].to_list(), yerr=xc['Yield_std'].to_list(), color=clrlist[xc[columnName].values])

        plt.xticks(xdata, labels, rotation='vertical')
        plt.ylabel('Yield [t/ha]')
        plt.title(crop_name)
        plt.subplots_adjust(bottom=0.25)
        plt.subplots_adjust(left=0.15)
        strFn = dirOutCluster + project['AOI'] + '_' + crop_name + '_90prctYield_ordered.png'
        plt.savefig(strFn.replace(" ", ""))
        plt.close()

        xc.sort_values(by=[columnName]).to_csv(dirOutCluster + '/' + crop_name + '_clusters.csv')

        # map it
        merged = map_df.merge(xc, how='left', left_on='AU_code', right_on="Region_ID")
        merged = merged.dropna(subset=['Region_ID'])
        # Production
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        listClustID = np.array(list(range(n_cluster_by_crop[c-1])))
        listColors = clrlist[listClustID]

        #make cluster a string
        color_mapping = dict(zip(listClustID, listColors))
        merged[columnName] = merged[columnName].astype(int)#.astype(str)
        merged.plot(color=merged[columnName].map(color_mapping), categorical=True, ax=ax, legend=True,
                    legend_kwds={'label': crop_name + ' Clusters', 'orientation': "horizontal"})
        #merged.plot(column=columnName, ax=ax, legend=True,
        #            legend_kwds={'label': crop_name + ' Clusters', 'orientation': "horizontal"})
        # Add Labels (only clustered regions)

        merged['coords'] = merged['geometry'].apply(lambda x: x.representative_point().coords[:])
        merged['coords'] = [coords[0] for coords in merged['coords']]
        for idx, row in merged.iterrows():
            plt.annotate(text=row['labels'], xy=row['coords'], horizontalalignment='center', fontsize=6, color='black')
        # plt.show()
        fig.savefig(
            dirOutCluster + project['AOI'] + '_' + crop_name + '_clusters.png')  # dirProcessed + '/' + project['AOI'] + '_90prctProd_PrctProd' + c + '.png')  # , dpi = 300)
        plt.close(fig)
        #y = stats[(stats['Region_ID'].isin(regionID_list)) & (stats['Crop_ID']==c)]
        #df = y[['Region_ID','AU_name','Yield']]
        #cropname = y['Crop_name'].iloc[0]
        #df.to_csv(dirOut + '/' + project['AOI'] + cropname + '_4HSD.csv', index=False)
        #plt.bar(xdata, xc['Yield', 'mean'].to_list(), yerr=xc['Yield', 'std'].to_list())

        print('debug')

def attempt_cluster_based_on_regression(target):
    pd.set_option('display.max_columns', None)
    desired_width = 400
    pd.set_option('display.width', desired_width)

    project = b05_Init.init(target)
    dirOut = project['output_dir']
    dirOutCluster = dirOut + '/AU_clusters'
    dirOutClusterP = Path(dirOutCluster)
    dirOutClusterP.mkdir(parents=True, exist_ok=True)

    dirAmdUn = project['input_dir'] + '/AdminUnits'
    # load shp
    fp = dirAmdUn + '/' + project['AOI'] + '_AdminUnits.shp'
    map_df = gpd.read_file(fp)

    # open stats and predictors
    stats90 = pd.read_pickle(dirOut + '/' + project['AOI'] + '_stats90.pkl')
    stats = pd.read_pickle(dirOut + '/' + project['AOI'] + '_stats.pkl')
    timerange = project['timeRange']
    stats = stats.drop(stats[stats['Year'] < timerange[0]].index)
    features = pd.read_pickle(dirOut + '/' + project['AOI'] + '_pheno_features4scikit.pkl')
    n_cluset_by_crop = [4,6,5]
    # For each crop (1,2,3) get the 90% regions
    for c in range(1, 4):
        regionID_list = stats90[stats90['Crop_ID'] == c]['Region_ID']
        region_names = stats90[stats90['Crop_ID'] == c]['AU_name']
        crop_name = stats90.loc[stats90['Crop_ID'] == c].iloc[0]['Crop_name'].values[0]
        print('Crop ' + str(c) + ', ' + crop_name)
        #print(region_names)

        y = stats[(stats['Region_ID'].isin(regionID_list)) & (stats['Crop_ID'] == c)]
        # y.head()
        y = y.drop(['ASAP1_ID'], axis=1)
        y = y.drop(['AU_name'], axis=1)
        y = y.drop(['Region_ID'], axis=1)
        z = pd.merge(y, features, how='left', left_on=['AU_code', 'Year'], right_on=['AU_code', 'YearOfEOS'])
        #print(z.head())

        # compute mean ND, T, P and Rad over the whole period
        P_cols = [col for col in z.columns if 'NDM' in col] # get M sampling
        z['ND_avg'] = z[P_cols].mean(axis=1)
        P_cols = [col for col in z.columns if 'TM' in col]  # get M sampling
        z['T_avg'] = z[P_cols].mean(axis=1)
        P_cols = [col for col in z.columns if 'RadM' in col]  # get M sampling
        z['Rad_avg'] = z[P_cols].mean(axis=1)
        P_cols = [col for col in z.columns if 'RainSumM' in col]  # get M sampling
        z['RainM_avg'] = z[P_cols].mean(axis=1)
        # determine multiple correlation coeff bu AUs
        columns = ['au_name','Region_ID', 'offset', 'gND', 'gT', 'gRad', 'gRain']
        df = pd.DataFrame(columns=columns)
        for au, id in zip(region_names['first'].to_list(), regionID_list.values.tolist()):
            #print(au)
            X = z[z['AU_name']==au][['ND_avg','T_avg','Rad_avg','RainM_avg']].to_numpy()
            y = z[z['AU_name']==au]['Yield'].to_numpy()
            reg = LinearRegression().fit(X, y)
            score = reg.score(X, y)
            rowDf = pd.DataFrame([[au] + [id] + np.append(reg.intercept_, reg.coef_).tolist()], columns=columns)
            df = df.append(rowDf)
        # cluster the coeff
        # keep only relevant variables
        data = df[['offset', 'gND', 'gT', 'gRad', 'gRain']].values
        #data.shape
        #print(data[0, :])
        plt.figure(figsize=(10, 7))
        plt.title("Country Dendograms")
        plt.xlabel("Progressive number")
        dend = shc.dendrogram(shc.linkage(data, method='ward'))
        plt.show()
        cluster = AgglomerativeClustering(n_clusters=n_cluset_by_crop[c-1], \
                                          affinity='euclidean', linkage='ward')
        groups = cluster.fit_predict(data)
        columnName = 'Ward_' + str(n_cluset_by_crop[c-1]) + 'groups'
        df[columnName] = groups
        df.sort_values(by=[columnName]).to_csv(dirOutCluster + '/' + crop_name + '_clusters.csv')
        # map it
        merged = map_df.merge(df, how='left', left_on="AU_code", right_on="Region_ID")
        # Production
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        merged.plot(column=columnName, ax=ax, legend=True,
                    legend_kwds={'label':crop_name + ' Clusters', 'orientation': "horizontal"})
        # Add Labels (only clustered regions)
        merged = merged.dropna(subset=['Region_ID']) #merged[merged['Crop_name'] == c]
        merged['coords'] = merged['geometry'].apply(lambda x: x.representative_point().coords[:])
        merged['coords'] = [coords[0] for coords in merged['coords']]
        for idx, row in merged.iterrows():
            plt.annotate(text=row['au_name'], xy=row['coords'], horizontalalignment='center', fontsize=6, color='r')
        # plt.show()
        fig.savefig(dirOutCluster + '/' + crop_name + '_clusters.png') #dirProcessed + '/' + project['AOI'] + '_90prctProd_PrctProd' + c + '.png')  # , dpi = 300)
        plt.close(fig)
        print('debug')