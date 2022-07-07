import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as scistats
import seaborn as sns
import matplotlib.pyplot as plt

import b05_Init

def explore(target):
    pd.set_option('display.max_columns', None)
    desired_width = 400
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)


    project = b05_Init.init(target)
    prct2retain = project['prct2retain']
    dirOut = project['output_dir']
    dirOutExplore = dirOut + '/Data_exploration'
    dirOutExplore = Path(dirOutExplore)
    dirOutExplore.mkdir(parents=True, exist_ok=True)
    #open stats and predictors
    statsX = pd.read_pickle(dirOut + '/' + project['AOI'] + '_stats'+ str(prct2retain) + '.pkl')
    stats = pd.read_pickle(dirOut + '/' + project['AOI'] + '_stats.pkl')
    timerange = project['timeRange']
    stats = stats.drop(stats[stats['Year'] < timerange[0]].index)
    features = pd.read_pickle(dirOut + '/' + project['AOI'] + '_pheno_features4scikit.pkl')

    # For each crop (1,2,3) get the 90% regions
    cs = statsX['Crop_ID', ''].unique()
    for c in cs:
        regionID_list = statsX[statsX['Crop_ID'] == c]['Region_ID']
        region_names = statsX[statsX['Crop_ID'] == c]['AU_name']
        crop_name = statsX.loc[statsX['Crop_ID'] == c].iloc[0]['Crop_name']
        print('Crop ' + str(c) + ', ' + crop_name)
        print(region_names)

        y = stats[(stats['Region_ID'].isin(regionID_list)) & (stats['Crop_ID']==c)]
        #y.head()
        y = y.drop(['ASAP1_ID'], axis=1)
        y = y.drop(['AU_name'], axis=1)
        y = y.drop(['Region_ID'], axis=1)
        z = pd.merge(y,features,how='left',left_on=['AU_code','Year'], right_on=['AU_code','YearOfEOS'])
        #print(z.head())

        # check if soil moisture is available
        if 'SMP1' in z.columns:
            df_data = z[['Area', 'Yield', 'Production', 'NDP1', 'NDminP1', 'NDmaxP1',
                         'NDP2', 'NDminP2', 'NDmaxP2',
                         'NDP3', 'NDminP3', 'NDmaxP3',
                         'RadP1', 'RadP2', 'RadP3',
                         'RainSumP1', 'RainSumP2', 'RainSumP3', 'TP1', 'TminP1', 'TmaxP1', 'TP2', 'TminP2',
                         'TmaxP2', 'TP3', 'TminP3', 'TmaxP3',
                         'SMP1', 'SMP2', 'SMP3']]
        else:
            df_data = z[['Area', 'Yield', 'Production', 'NDP1', 'NDminP1', 'NDmaxP1',
                         'NDP2', 'NDminP2', 'NDmaxP2',
                         'NDP3', 'NDminP3', 'NDmaxP3',
                         'RadP1', 'RadP2', 'RadP3',
                         'RainSumP1', 'RainSumP2', 'RainSumP3', 'TP1', 'TminP1', 'TmaxP1', 'TP2', 'TminP2',
                         'TmaxP2', 'TP3', 'TminP3', 'TmaxP3']]
        # correlation
        correlation_data = df_data.corr()
        # p-value
        def corr_sig(df=None):
            p_matrix = np.zeros(shape=(df.shape[1], df.shape[1]))
            for col in df.columns:
                for col2 in df.drop(col, axis=1).columns:
                    x = df[col].values
                    y = df[col2].values
                    nas = np.logical_or(np.isnan(x), np.isnan(y))
                    #corr = sp.pearsonr()
                    _, p = scistats.pearsonr(x[~nas], y[~nas]) # , nan_policy='omit'
                    p_matrix[df.columns.to_list().index(col), df.columns.to_list().index(col2)] = p
            return p_matrix

        p_values = corr_sig(df_data)
        mask = np.invert(np.tril(p_values < 0.05))
        # note seaborn will hide correlation were the boolean value is True in the mask

        #mask = np.zeros_like(correlation_data, dtype=np.bool)
        #mask[np.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=(11, 9))
        plt.show(block = False)
        # Generate a custom diverging colormap
        cmap = sns.palette = "vlag"

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(correlation_data, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5});
        plt.title(crop_name.values[0], fontsize=20, y=1)
        t = ', '.join(region_names.values.flatten())
        # plt.text(0.7, 0.93, t, ha='left', wrap=True, fontsize=10,  transform=ax.transAxes)
        fn = project['AOI'] + '_'+crop_name.values[0]+'_corr.png'
        plt.savefig(dirOutExplore / fn)
        plt.close()
        #loop on pheno phase to show paiered plots
        for pp in range(1,4,1):
            #f, ax = plt.subplots(figsize=(11, 9))
            columns = df_data.columns
            #extract pp phase
            columns_pp=[s for s in columns if str(pp) in s]
            for s in ['Area', 'Yield', 'Production']:
                columns_pp.insert(0, s)
            df_data_pp = df_data[columns_pp]
            df_data_pp.loc[:]['Production']= df_data_pp['Production'].div(1000)
            df_data_pp.loc[:]['Area'] = df_data_pp['Area'].div(1000)
            rad_cols = [col for col in df_data_pp.columns if 'Rad' in col]
            df_data_pp.loc[:][rad_cols] = df_data_pp[rad_cols].div(1000)
            sns.set(font_scale=0.75)
            sns.pairplot(data=df_data_pp,height=1,plot_kws={"s": 10},corner=True)
            plt.subplots_adjust(bottom=0.05)
            plt.subplots_adjust(left=0.05)
            fn = project['AOI'] + '_' + crop_name.values[0] + '_phase'+str(pp)+'_paired_corr.png'
            plt.savefig(dirOutExplore / fn)
            plt.close()
        print('ok')

