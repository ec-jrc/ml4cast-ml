import pandas as pd
import os
import src.constants as cst
import preprocess.b1000_preprocess_utilities as b1000_preprocess_utilities
import seaborn as sns
import matplotlib.pyplot as plt
import b05_Init

'''
This function is just checking for NaN in all data (predictors and stats)
'''

def check(target):
    project = b05_Init.init(target)
    prct2retain = project['prct2retain']    #'_stats'+ str(prct2retain) + '.pkl'
    statsX = pd.read_pickle(os.path.join(cst.odir, target,
                                          f'{target}_stats{str(prct2retain)}.pkl'))  # TODO read_pickle  # major AU #TODO: make sure file paths are harmonised across local and condor machines
    stats = pd.read_pickle(os.path.join(cst.odir, target, f'{target}_stats.pkl'))  # all targetCountry stats
    # drop years before period of interest
    stats = stats.drop(stats[stats['Year'] < project['timeRange'][0]].index)
    # drop unnecessary column
    stats = stats.drop(cst.drop_cols_stats, axis=1)
    # rescale Production units for better graphic and better models
    # stats['Production'] = stats['Production'].div(cst.production_scaler)

    # open predictors
    raw_features = pd.read_pickle(os.path.join(cst.odir, target, f'{target}_pheno_features4scikit.pkl'))

    # merge stats and features, so that at ech stat entry I have the full set of features
    yxData = pd.merge(stats, raw_features, how='left', left_on=['AU_code', 'Year'],
                      right_on=['AU_code', 'YearOfEOS'])
    cs = statsX['Crop_ID',''].unique()
    for c in cs:
        yxDatac = b1000_preprocess_utilities.retain_X(yxData, statsX, c)
        print(yxDatac['Crop_name'].iloc[0])
        if len(yxDatac[yxDatac['Yield'].isnull()]) !=0:
            print('Nan in yield')
            print(yxDatac[yxDatac['Yield'].isnull()])
        if len(yxDatac[yxDatac['Area'].isnull()]) != 0:
            print('Nan in Area:')
            print(yxDatac[yxDatac['Area'].isnull()])
            yxDatac[yxDatac['Area'].isnull()]
        doplot = 0
        if yxDatac.isnull().values.any():
            doplot = 1
            print('Nan id the df, check!')
        if doplot ==1:
            g = sns.relplot(
                data=yxDatac,
                x="Year", y="Yield", col="AU_name", #hue="year",
                kind="line", palette="crest", linewidth=2, zorder=5,
                col_wrap=3, height=1, aspect=2, legend=False,
            )
            g.tight_layout()
            plt.show()
            doplot = 0
        print('**************')
    print('ended')