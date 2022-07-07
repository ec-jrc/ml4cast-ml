import numpy as np
import pandas as pd
import pandasql as ps
import pymannkendall as mk
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.linear_model import TheilSenRegressor
from scipy import stats
import os
import b05_Init
import src.constants as cst


def trend_analysis(target):
    project = b05_Init.init(target)
    prct2retain = project['prct2retain']
    dir = project['output_dir']
    # ensamble option: all the graphs on the same plot
    ensamble_option = False
    # trend in the graph option
    trend_option = True
    # test trend concept (use year before/after in the firstYearPredictors 2002 on), limited to ny
    test_concept = False
    ny = cst.ny_max_trend

    # test trend forward with fixed length
    test_forward_trend = True
    # limit oldest year
    limit_oldest = False
    if limit_oldest == True:
        firstYearStat = 1983
    firstYearPredictors = project['timeRange'][0]


    # input paramters ---
    # input statistics
    file = dir + '/' + project['AOI'] + '_stats.pkl'
    # input 90% statistics
    file_main_regions = dir + '/' + project['AOI'] + '_stats'+ str(prct2retain) + '.pkl'
    # test significance
    alpha = 0.01
    # output folder
    output_folder = dir + '/' + '/ManKendallTest'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    #load units
    dirStat = project['input_dir'] + '/CountryStats'
    units = pd.read_csv(dirStat + '/' + project['AOI'] + '_measurement_units.csv')
    #area_unit = units['Area'].values[0]
    yield_unit = units['Yield'].values[0]




    # intialize the output dataframe
    out_name = 'MK_test_' + str(alpha) + '.csv'
    MK_list = []
    MK_columns = ['AU_name', 'Crop_name', 'trend', 'h', 'p value', 'z', 'Tau', 's', 'var_s', 'slope', 'intercept']
    #initialize color form matplotlib
    color = 0
    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = prop_cycle.by_key()['color']



    #read the data
    statistics = pd.read_pickle(file)
    statistics_90 = pd.read_pickle(file_main_regions)
    # drop unncessary columns
    statistics_90= statistics_90.drop([('Production','std'),('Yield','std'),('Area','std')], axis=1)
    statistics_90.columns = statistics_90.columns.droplevel(1)
    #print (statistics.head())
    region_crop = statistics_90.groupby(['AU_name', 'Crop_name']).size().reset_index()
    #limit years
    if limit_oldest == True:
        statistics = statistics[statistics['Year'] >= firstYearStat]
    else:
        firstYearStat = statistics['Year'].min()

    trendwindow = np.min([firstYearPredictors - firstYearStat, ny])

    #print (region_crop)
    #-------------------------
    # Drop years if needed
    # indexNames = statistics[statistics['Year'] == 2001].index
    # Delete these row indexes from dataFrame
    # statistics.drop(indexNames, inplace=True)
    #-------------------------


    if ensamble_option == True:
        #mpl.style.use('seaborn')
        fig, ax = plt.subplots(figsize=(12, 8))

    for i in range (0, region_crop.shape[0]):
        #i = 11
        crop = region_crop['Crop_name'].iloc[i]
        region = region_crop['AU_name'].iloc[i]
        print (i, crop, region)
        # statistics.loc[(statistics['Crop_name'] == crop) &
        #     #                (statistics['AU_name'] == region)]

        yield_data_df = statistics[['Year', 'Yield']].loc[(statistics['Crop_name'] == region_crop['Crop_name'].iloc[i]) &
                                                          (statistics['AU_name'] == region_crop['AU_name'].iloc[
                                                              i])].copy().set_index('Year')
        LastYearStat = yield_data_df.index.max()
        # execute the test and compute the trend
        # Michele: the mk estimation of mk is correct and keep track of missing values, proof with cipy belo
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(yield_data_df, alpha=alpha)
        trend_line = np.arange(len(yield_data_df)) * slope + intercept
        # trend_line ST with scikit to handle missing does not work, there is a bu posted on stack   https://stackoverflow.com/questions/67414818/theil-sen-regression-different-results-when-translating-x-axis

        y = yield_data_df.values.reshape(-1)
        y0 = np.copy(y)
        X = yield_data_df.index.values

        X0 = np.copy(X)
        X= X[~np.isnan(y)].reshape(-1,1)
        #X = np.arange(len(yield_data_df))
        # #X2 = X2[~np.isnan(y)].reshape(-1, 1)
        # reg = TheilSenRegressor(random_state=0).fit(X, y)
        # reg2 = TheilSenRegressor(random_state=0).fit(X2, y)
        # slope_sk = reg.coef_
        # intercept_sk = reg.intercept_
        y = y[~np.isnan(y)]

        res = stats.theilslopes(y, X)
        resFit = res[1] + res[0] * X
        # scikit
        # X = np.arange(len(yield_data_df))
        # #X2 = X2[~np.isnan(y)].reshape(-1, 1)
        # reg = TheilSenRegressor(random_state=0).fit(X, y)
        # reg2 = TheilSenRegressor(random_state=0).fit(X2, y)
        # slope_sk = reg.coef_
        # intercept_sk = reg.intercept_

        #--------------------------------------------------------------------
        # plot data and trend
        if ensamble_option == False:
            #mpl.style.use('seaborn')
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(yield_data_df)
            if trend_option == True:
                ax.plot(yield_data_df.index, trend_line, label= F'Theil-Sen trend line')
                ax.plot(X0,  res[1] + res[0] * X0, label=F'Theil-Sen scipy', color='r')
                if test_concept:
                    X00 = np.copy(X0)
                    y00 = np.copy(y0)
                    hl = round(len(X00)/2)
                    X00 = X00[0:hl]
                    y00 = y00[0:hl]
                    trend00, h00, p00, z00, Tau00, s00, var_s00, slope00, intercept00 = mk.original_test(y00, alpha=alpha)
                    X00 = X00[~np.isnan(y00)].reshape(-1, 1)
                    y00 = y00[~np.isnan(y00)]
                    res = stats.theilslopes(y00, X00)
                    ax.plot(X0[0:hl], res[1] + res[0] * X0[0:hl], label=F'Theil-Sen up to middle', color='b')
                    ax.plot(X0[hl], res[1] + res[0] * X0[hl], label=F'Theil-Sen up to middle', color='b',  marker='o', linestyle='')
                if test_forward_trend == True:
                    for i in np.arange(firstYearPredictors-1, LastYearStat+1):
                        X00 = np.copy(X0)
                        y00 = np.copy(y0)
                        indexes = np.where(np.logical_and(X00>=i-trendwindow, X00<=i))
                        X00 = X00[indexes]
                        y00 = y00[indexes]
                        #print(X00)
                        X00 = X00[~np.isnan(y00)].reshape(-1, 1)
                        y00 = y00[~np.isnan(y00)]
                        res = stats.theilslopes(y00, X00)
                        ax.plot(X00, res[1] + res[0] * X00, color='b', linewidth=0.2)
                        ax.plot(i+1, res[1] + res[0] * (i+1), color='b', marker='o', linestyle='')

            ax.set_xlabel('Years')
            ax.set_ylabel('Yield [' + yield_unit + ']')
            ax.set_xlim(left=yield_data_df.index.min(), right=yield_data_df.index.max())
            #ax.set_ylim(bottom=0, top=4.5)
            ax.set_xticks(yield_data_df.index)
            ax.set_xticklabels(yield_data_df.index)
            ax.set_xticklabels(ax.get_xticks(), rotation = 90)
            if test_concept:
                fig.suptitle(F'{crop}' + ' - ' + F'{region}' + ', p=' + F'{round(p, 3)}' +  ', p_middle=' + F'{round(p00, 3)}', fontsize=20, fontweight='bold')
            else:
                fig.suptitle(F'{crop}' + ' - ' + F'{region}' + ', p=' + F'{round(p, 3)}', fontsize=20,
                             fontweight='bold')
            plt.legend(loc='best')#, bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True, ncol=5)
            fn_out = output_folder + '/' + crop + '_' + region + '.png'
            plt.savefig(fn_out, bbox_inches='tight')
            plt.close()
        else:
            ax.plot(yield_data_df, color = 'C' + str(color), label=F'{crop} - {region}')
            if trend_option == True:
                ax.plot(yield_data_df.index, trend_line, color = 'C' + str(color), label= F'{crop} trend line')

        # change color
        color = color + 1
        # plt.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.show()

        # --------------------------------------------------------------------
        MK_list.append([region, crop, trend, h, p, z, Tau, s, var_s, slope, intercept])
        #print (MK_list)
    #    ax.legend(['data', ' Theil-Sen trend line'])
    if ensamble_option == True:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=5)
        ax.set_xlabel('Years')
        ax.set_ylabel('Yield [' + yield_unit + ']')
#        ax.set_ylabel('Yield [t/ha]')
        ax.set_xticks(yield_data_df.index)
        ax.set_xticklabels(yield_data_df.index)
        fig.suptitle(F'{region}', fontsize=20, fontweight='bold')
        fn_out = output_folder + '/' + crop + '_' + region + '.png'
        plt.savefig(fn_out, dpi=600)
    plt.close()
    MK_df = pd.DataFrame(MK_list, columns=MK_columns)
    MK_df.to_csv(output_folder + '\\' + out_name, index=False, header=True)