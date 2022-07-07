import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import seaborn as sns
from pathlib import Path
import b05_Init
from preprocess import phenology
import copy
import os



def plots_and_pheno(target):
    '''Compute the pheno and save plots, work only on largest 90% wilayas'''
    pd.set_option('display.max_columns', None)
    desired_width=320
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns',10)

    project = b05_Init.init(target)
    prct2retain = project['prct2retain']
    dirStat = project['input_dir'] + '/CountryStats'
    dirOut =  project['output_dir']
    dirProcessed = project['output_dir']
    dirOutRP = dirOut + '/RegionProfiles'
    dirOutRP = Path(dirOutRP)
    dirOutRP.mkdir(parents=True, exist_ok=True)

    x = pd.read_pickle(dirProcessed + '/' + project['AOI'] +'_predictors.pkl')
    x = x[x['variable_name'] == 'NDVI']
    x = x.rename(columns={'mean': 'NDVI'})
    #print(x.head())
    stats = pd.read_pickle(dirProcessed + '/' + project['AOI'] + '_stats'+ str(prct2retain) + '.pkl')
    # get available Wilayas
    uniqueAuNames = stats[('AU_name','first')].unique()
    n = uniqueAuNames.size

    # look for a file pescribing the modality
    if os.path.exists(dirProcessed + '/' + project['AOI'] + '_modality.csv'):
        print('****************************************************************')
        print('Warning b40: Modality from file')
        print('****************************************************************')
        mod = pd.read_csv(dirProcessed + '/' + project['AOI'] + '_modality.csv')
    else:
        print('****************************************************************')
        print('Warning b40: The modality file does not exist, monomodality assumed')
        print('****************************************************************')
        mod = pd.DataFrame(columns=['AU_name','gs_per_year','season_id_for_mono','dek0_start1','dek0_end1','dek0_start2','dek0_end2'])
        #season_id_for_mono indicates if the only season is to be considered season 1 or season 2 of bimodal
        #dek0 start and end refers to the periods to be considered for the bimodal profile (align with the monomodal, either season 1 or 2)
        # when mono and bi seasons are present, the bimodal season aligned with the mono must stay in period 1
        mod['AU_name'] = uniqueAuNames
        mod['gs_per_year'] = 1
        mod['season_id_for_mono'] = 1
        mod[['dek0_start1','dek0_end1','dek0_start2','dek0_end2']] = None
        mod.to_csv(dirProcessed + '/' + project['AOI'] + '_modality.csv')

    colors = pl.cm.jet(np.linspace(0,1,n))
    #fig, ax = plt.subplots()
    hndlr = list()
    for i in range(n):
        xW = x[x['AU_name'] == uniqueAuNames[i]]
        #plt.plot_date(xW.loc[:, 'time'], xW.loc[:, 'nd'], color=colors[i], marker='',linestyle='-', linewidth=0.2, label=uniqueAuNames[i])
        plt.plot_date(xW['Date'], xW['NDVI'], color=colors[i], marker='None', linestyle='-', linewidth=0.2, label=uniqueAuNames[i])
    #print(hndlr)
    plt.legend(loc='best', fontsize='xx-small')
    fn = project['AOI'] + '_All_AUs.png'
    plt.savefig(dirOutRP / fn)
    plt.close()

    # create an empty df to store pheno results
    #phenoDf = pd.DataFrame(columns=['AU_code','AU_name','season','SOS','TOM','SEN','EOS','SOS36','TOM36','SEN36','EOS36'])
    phenoDf = pd.DataFrame(columns=['AU_code', 'AU_name', 'season', 'SOS', 'TOM', 'SEN', 'EOS', 'SOS108', 'TOM108', 'SEN108', 'EOS108'])
    #save a plot per region
    if True:
        dfLine = 0
        for i in range(n):
            xW = x[x['AU_name'] == uniqueAuNames[i]]
            print(uniqueAuNames[i])
            if mod[mod['AU_name'] == uniqueAuNames[i]]['gs_per_year'].values[0] == 1: #x.loc[x['A'] == 2, 'B']
                gspy = 1
                startRangeDekFrom0 = [None, None]
                endRangeDekFrom0 = [None, None]
                season = [mod[mod['AU_name'] == uniqueAuNames[i]]['season_id_for_mono'].values[0], mod[mod['AU_name'] == uniqueAuNames[i]]['season_id_for_mono'].values[0]]
            else:
                gspy = 2
                startRangeDekFrom0 = [mod[mod['AU_name'] == uniqueAuNames[i]]['dek0_start1'].values[0], mod[mod['AU_name'] == uniqueAuNames[i]]['dek0_start2'].values[0]]
                endRangeDekFrom0 =  [mod[mod['AU_name'] == uniqueAuNames[i]]['dek0_end1'].values[0], mod[mod['AU_name'] == uniqueAuNames[i]]['dek0_end2'].values[0]]
                season = [1,2]
            #codeCurrentAU = x.loc[x['AU_name'] == uniqueAuNames[i]].iloc[0]['AU_code']
            codeCurrentAU = xW['AU_code'].iloc[0]
            #compute lta and sd
            lta = xW.groupby(xW.dek).mean()['NDVI']
            sd  = xW.groupby(xW.dek).std()['NDVI']
            dek = lta.index#.tolist()
            #doy = xW.groupby(xW.compInd).mean()['DOY']
            #dek = xW.groupby(xW.dek).mean()['dek']
            #fig, ax = plt.subplots(2,1,1)
            plt.subplot(2, 1, 1)
            plt.plot_date(xW['Date'], xW['NDVI'], color='blue', marker='',linestyle='-', linewidth=0.5, label=uniqueAuNames[i])
            plt.xlabel('Date')
            plt.ylabel('NDVI')
            plt.title(uniqueAuNames[i])
            plt.subplot(2, 1, 2)
            #get the time in doy
            plt.plot(dek, lta, color='blue', marker='', linestyle='-', linewidth=0.5, label=uniqueAuNames[i])
            plt.plot(dek, lta - sd, color='blue', marker='', linestyle='--', linewidth=0.2, label=uniqueAuNames[i])
            plt.plot(dek, lta + sd, color='blue', marker='', linestyle='--', linewidth=0.2, label=uniqueAuNames[i])
            #compute pheno
            for s in range(gspy):
                pheno = phenology.pheno_lta_dek_mono_wrapper_v2(lta.values, dek.values,
                                                                prctSOS=project['prctSOS'], prctSEN=project['prctSEN'], prctEOS=project['prctEOS'], startRangeDekFrom0=startRangeDekFrom0[s], endRangeDekFrom0=endRangeDekFrom0[s]) # panda series to array
                pheno36 = copy.deepcopy(pheno)
                for t in ('SOS', 'SEN','EOS', 'TOM'):
                    if pheno36[t] > 36:
                        pheno36[t] = pheno36[t] - 36
                #phenoDf = pd.DataFrame(columns=['AU_code', 'AU_name', 'season', 'SOS', 'TOM', 'SEN', 'EOS', 'SOS108', 'TOM108', 'SEN108', 'EOS108'])
                phenoDf.loc[dfLine] = [codeCurrentAU, uniqueAuNames[i], season[s], pheno36['SOS'], pheno36['TOM'], pheno36['SEN'], pheno36['EOS'],
                                       pheno['SOS'], pheno['TOM'], pheno['SEN'], pheno['EOS'], ]
                #phenoDf.loc[dfLine] = [codeCurrentAU, uniqueAuNames[i], season[s], pheno['SOS'], pheno['TOM'], pheno['SEN'],pheno['EOS'],pheno36['SOS'], pheno36['TOM'], pheno36['SEN'],pheno36['EOS']]
                dfLine = dfLine + 1
                axes = plt.gca()
                yrange = axes.get_ylim()
                plt.plot([pheno36['TOM'], pheno36['TOM']], [yrange[0], 1], color='yellow', marker='', linestyle='--')
                plt.plot([pheno36['SOS'], pheno36['SOS']], [yrange[0], 1], color='green', marker='', linestyle='--')
                plt.plot([pheno36['SEN'], pheno36['SEN']], [yrange[0], 1], color='blue', marker='', linestyle='--')
                plt.plot([pheno36['EOS'], pheno36['EOS']], [yrange[0], 1], color='red', marker='', linestyle='--')
                axes.set_ylim(yrange)
                plt.xlabel('Dekad')
                plt.ylabel('NDVI')
                plt.tight_layout()
            fn = project['AOI'] + '_' + uniqueAuNames[i] +'.png'
            plt.savefig(dirOutRP / fn)
            #plt.show()
            plt.close()


        dirOutPheno = dirOut  + '/Pheno'
        dirOutPheno = Path(dirOutPheno)
        dirOutPheno.mkdir(parents=True, exist_ok=True)
        phenoDf[["SOS", "TOM", "SEN", "EOS"]] = phenoDf[["SOS", "TOM", "SEN", "EOS"]].apply(pd.to_numeric)
        fn = project['AOI'] + '_pheno.pkl'
        phenoDf.to_pickle(dirOutPheno / fn)
        fn = project['AOI'] +'_pheno.csv'
        phenoDf.to_csv(dirOutPheno / fn)

        for s in range(gspy):
            if gspy == 1:
                fn = project['AOI'] + '_pheno_stats.csv'
            else:
                fn = project['AOI'] + '_pheno_stats_season' + str(s + 1) + '.csv'
            tmp = phenoDf[phenoDf['season'] == s + 1]
            tmp.describe().to_csv(dirOutPheno / fn, index_label="Stats")

            fig, axs = plt.subplots(nrows=4)
            plt.show(block=False)
            x = tmp.SOS.to_numpy()
            sns.distplot(x, bins=range(1, 38, 2), kde=False, rug=True, ax=axs[0]);
            axs[0].set(xlabel='Dekad', ylabel='SOS n')  # title='some title'
            axs[0].minorticks_on()

            x = tmp.TOM.to_numpy()
            sns.distplot(x, bins=range(1, 38, 2), kde=False, rug=True, ax=axs[1]);
            axs[1].set(xlabel='Dekad', ylabel='TOM n')

            x = tmp.SEN.to_numpy()
            sns.distplot(x, bins=range(1, 38, 2), kde=False, rug=True, ax=axs[2]);
            axs[2].set(xlabel='Dekad', ylabel='SEN n')

            x = tmp.EOS.to_numpy()
            sns.distplot(x, bins=range(1, 38, 2), kde=False, rug=True, ax=axs[3]);
            axs[3].set(xlabel='Dekad', ylabel='EOS n')

            plt.subplots_adjust(top=0.92, bottom=0.12, left=0.10, right=0.95, hspace=0.5, wspace=0.35)
            if gspy == 1:
                fn = project['AOI'] + '_pheno_histo.png'
            else:
                fn = project['AOI'] + '_pheno_histo_season' + str(s + 1) + '.png'
            plt.savefig(dirOutPheno / fn)

