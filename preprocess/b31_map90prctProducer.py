import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import b05_Init

def map90prctProducer(target):
    admin0_field_in_shp = 'adm0_name' #sometime 'name0'
    pd.set_option('display.max_columns', None)
    desired_width=320
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    if target == 'ZAsummer':
        name0 = 'South Africa'
    else:
        name0 = target


    project = b05_Init.init(target)
    prct2retain = project['prct2retain']
    #time period

    #dirAmdUn =  project['input_dir'] + '/AdminUnits'
    dirAmdUn = project['input_dir'] + '/ASAPunits/gaul1_asap_v04'
    dirProcessed = project['output_dir']

    #https://towardsdatascience.com/a-beginners-guide-to-create-a-cloropleth-map-in-python-using-geopandas-and-matplotlib-9cc4175ab630

    #load shp
    fp = dirAmdUn + '/gaul1_asap.shp'
    map_df = gpd.read_file(fp)
    # check the GeoDataframe
    #print(map_df.head())

    #load largest producers
    suffix = '_stats' + str(prct2retain)
    stats = pd.read_pickle(dirProcessed + '/' + project['AOI'] + suffix + '.pkl')
    #print(stats.head())

    #merge on crop
    uniqueCrops = stats[('Crop_name','first')].unique()
    nCrop = uniqueCrops.size
    #colors = pl.cm.jet(np.linspace(0,1,n))

    for c in uniqueCrops:
        statsCrop = stats[stats[('Crop_name','first')] == c]
        #remove level 1
        statsCrop = statsCrop.rename(columns={'first': ''})
        statsCrop.columns = [''.join(col) for col in statsCrop.columns]
        #print(statsCrop.head())
        #join the two levels
        #merged = map_df.merge(statsCrop, how='left', left_on="AU_code", right_on="Region_ID")
        map_df = map_df[map_df[admin0_field_in_shp] == name0]
        merged = map_df.merge(statsCrop, how='left', left_on="asap1_id", right_on="Region_ID")
        merged = merged[['AU_name', 'geometry', 'Crop_name', 'Yieldmean', 'Productionmean', 'Perc_production','Areamean']]
        #print(merged.head())
        #Production
        fig, ax = plt.subplots(1, 1, figsize=(10,5))
        merged.plot(column='Perc_production', ax=ax, legend=True, legend_kwds={'label': c+' % national production','orientation': "horizontal", 'shrink': 0.4}, vmin=1, vmax=11)
        # Add Labels (only plotted regions
        merged = merged[merged['Crop_name'] == c]
        merged['coords'] = merged['geometry'].apply(lambda x: x.representative_point().coords[:])
        merged['coords'] = [coords[0] for coords in merged['coords']]
        for idx, row in merged.iterrows():
            plt.annotate(text=row['AU_name'], xy=row['coords'], horizontalalignment='center', fontsize=6, color='r')
        #plt.show()
        plt.ylabel('Deg N')
        plt.xlabel('Deg E')
        start, end = plt.xlim()
        ss = np.floor(start)
        ee = np.ceil(end)
        plt.xlim(ss,ee)
        plt.xticks(np.arange(ss, ee, 1))
        plt.xlim(start, end)

        start, end = plt.ylim()
        ss = np.floor(start)
        ee = np.ceil(end)
        plt.ylim(ss, ee)
        plt.yticks(np.arange(ss, ee, 1))
        plt.ylim(start, end)
        plt.tight_layout()

        suffix = '_map_' + str(prct2retain) + 'prctProd_PrctProd_'
        fig.savefig(dirProcessed + '/' + project['AOI'] + suffix +c+'.png')#, dpi = 300)
        plt.close(fig)

        # Area
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        lbl = r'${\rm \/ Area \/ (1000 km^2)}$'
        merged['Areamean'] = merged['Areamean']/1000
        merged.plot(column='Areamean', ax=ax, legend=True,
                    #legend_kwds={'label': c + ' Area (km2)', 'orientation': "horizontal", 'shrink': 0.4},
                    legend_kwds={'label': c + lbl, 'orientation': "horizontal", 'shrink': 0.4},
                    vmin=0, vmax=120)

        # Add Labels (only plotted regions)
        merged = merged[merged['Crop_name'] == c]
        merged['coords'] = merged['geometry'].apply(lambda x: x.representative_point().coords[:])
        merged['coords'] = [coords[0] for coords in merged['coords']]
        for idx, row in merged.iterrows():
            plt.annotate(text=row['AU_name'], xy=row['coords'], horizontalalignment='center', fontsize=6, color='r')
        # plt.show()
        plt.ylabel('Deg N')
        plt.xlabel('Deg E')
        start, end = plt.xlim()
        ss = np.floor(start)
        ee = np.ceil(end)
        plt.xlim(ss, ee)
        plt.xticks(np.arange(ss, ee, 1))
        plt.xlim(start, end)

        start, end = plt.ylim()
        ss = np.floor(start)
        ee = np.ceil(end)
        plt.ylim(ss, ee)
        plt.yticks(np.arange(ss, ee, 1))
        plt.ylim(start, end)
        plt.tight_layout()
        suffix = '_map_' + str(prct2retain) + 'prctProd_MeanArea_'
        fig.savefig(dirProcessed + '/' + project['AOI'] + suffix + c + '.png')  # , dpi = 300)
        plt.close(fig)

        # yield
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        merged.plot(column='Yieldmean', ax=ax, legend=True,
                    legend_kwds={'label': c + ' Mean yield', 'orientation': "horizontal"});
        # Add Labels (only plotted regions
        merged = merged[merged['Crop_name'] == c]
        merged['coords'] = merged['geometry'].apply(lambda x: x.representative_point().coords[:])
        merged['coords'] = [coords[0] for coords in merged['coords']]
        for idx, row in merged.iterrows():
            plt.annotate(text=row['AU_name'], xy=row['coords'], horizontalalignment='center', fontsize=6, color='r')
        # plt.show()
        suffix = '_map_' + str(prct2retain) + 'prctProd_MeanYield_'
        fig.savefig(dirProcessed + '/' + project['AOI'] + suffix + c + '.png')  # , dpi = 300)
        plt.close(fig)


# res = map90prctProducer('Algeria')