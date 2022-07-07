import pandas as pd
import numpy as np
import glob
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import b05_Init
import sklearn.metrics as metrics

def map_rRMSEp(dir, target, forecastTime):
    pd.set_option('display.max_columns', None)
    desired_width=320
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)


    metric2use = 'rRMSE_p'  # RMSE_p' #'R2_p'

    if metric2use == 'rRMSE_p':
        sortAscending = True
    else:
        print('The metric is not coded, the function cannot be executed')
        return -1



    project = b05_Init.init(target)
    #time period

    dirAmdUn =  project['input_dir'] + '/AdminUnits'
    dirProcessed = project['output_dir']

    #https://towardsdatascience.com/a-beginners-guide-to-create-a-cloropleth-map-in-python-using-geopandas-and-matplotlib-9cc4175ab630

    #load shp
    fp = dirAmdUn + '/' + project['AOI'] + '_AdminUnits.shp'
    map_df = gpd.read_file(fp)

    #get area stat anc compute % of crop c area over total crop
    # load largest producers
    # from stats I have the list of 90 % province (can be different by crop)
    stats90 = pd.read_pickle(dirProcessed + '/' + project['AOI'] + '_stats90.pkl')
    #stats90 = pd.read_csv(dirProcessed + '/' + project['AOI'] + '_stats90.csv')
    # to get the fraction area I have to load all provinces
    stats = pd.read_csv(dirProcessed + '/' + project['AOI'] + '_stats.csv')
    # get the mean annual area by crop
    x = stats.groupby(['AU_code','Crop_ID']). \
        agg({'AU_name': 'first', 'Crop_name': 'first', 'Area': ['mean'], 'Production': ['mean']})
    x.reset_index(inplace=True)
    x.columns = x.columns.get_level_values(0)
    #x.columns.droplevel(1)
    x = x.groupby(['AU_code']).agg({'AU_name': 'first', 'Area': ['sum']})
    x.reset_index(inplace=True)
    x.columns = x.columns.get_level_values(0)
    x.rename(columns={'Area': 'TotalArea'}, inplace=True)
    # now merge with stats 90
    cols = [5,6,7,8,10,11,12,13]
    stats90.drop(stats90.columns[cols], axis=1, inplace=True)
    stats90.columns = stats90.columns.get_level_values(0)
    stats90 = stats90.merge(x, left_on="Region_ID", right_on="AU_code")
    stats90['PercentArea'] = stats90['Area']/stats90['TotalArea']*100




    # get the best model at desired forecasting time for all crops
    bm = pd.read_csv(dir + '/' + 'all_model_output.csv')
    bm = bm[bm['forecast_time'] == forecastTime]
    crops = bm['Crop'].unique()
    for c in crops:
        bmc = bm[bm['Crop'] == c]
        if metric2use  == 'rRMSE_p':
            best_model_row = bmc.iloc[bmc[metric2use].argmin()]
        else:
            print('The metric is not coded, the function cannot be executed')
            return -1
        stats90c = stats90[stats90['Crop_name'] == c]
        run_id = best_model_row['runID']
        fn_res = glob.glob(os.path.join(dir, f'*{run_id}*_mRes.csv'))[0]
        mres = pd.read_csv(fn_res)
        # load its mRSE and compute error by province
        avg_y_true = mres['yLoo_true'].mean()
        df = mres.groupby('AU_code').apply(
            lambda x: metrics.mean_squared_error(x['yLoo_true'], x['yLoo_pred'], squared=False))
        df = df.to_frame('RMSE_p')
        df['rRMSE_p'] = df['RMSE_p'] / avg_y_true * 100.0
        # map it
        merged = map_df.merge(df, on="AU_code")
        merged= merged.merge(stats90c, on="AU_code")
        merged = merged[['AU_name', 'geometry', 'rRMSE_p', 'PercentArea']]
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        lbl = r'${\rm\/rRMSE_p\/(\%)}$'
        merged.plot(column='rRMSE_p', ax=ax, legend=True,
                    legend_kwds={'label': c + lbl, 'orientation': "horizontal"});
        # Add Labels (only plotted regions
        #merged = merged[merged['Crop_name'] == c]
        merged['coords'] = merged['geometry'].apply(lambda x: x.representative_point().coords[:])
        merged['coords'] = [coords[0] for coords in merged['coords']]
        for idx, row in merged.iterrows():
            plt.annotate(text=row['AU_name'], xy=row['coords'], horizontalalignment='center', fontsize=6, color='r')
        # plt.show()
        fig.savefig(dirProcessed + '/' + project['AOI'] + '_rRMSEp_map' + c + '.png')  # , dpi = 300)
        plt.close(fig)
        # print(merged.head())

        # now percenet area of crop
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        lbl = ', percent of total crop area (%)'
        merged.plot(column='PercentArea', ax=ax, legend=True,
                    legend_kwds={'label': c + lbl, 'orientation': "horizontal"});
        # Add Labels (only plotted regions
        # merged = merged[merged['Crop_name'] == c]
        merged['coords'] = merged['geometry'].apply(lambda x: x.representative_point().coords[:])
        merged['coords'] = [coords[0] for coords in merged['coords']]
        for idx, row in merged.iterrows():
            plt.annotate(text=row['AU_name'], xy=row['coords'], horizontalalignment='center', fontsize=6, color='r')
        # plt.show()
        fig.savefig(dirProcessed + '/' + project['AOI'] + '_percent_area_map' + c + '.png')  # , dpi = 300)
        plt.close(fig)


        #now look at correlation between rrmsep and % fraction

        print(c)
        print(merged['rRMSE_p'].corr(merged['PercentArea'], method='spearman'))




    #


res = map_rRMSEp('X:/PY_data/ML1_data_output/Algeria/Final_run/20210219','Algeria', 6) #6 is forecasting month of May