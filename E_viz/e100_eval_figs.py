import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pathlib
from D_modelling import d090_model_wrapper, d140_modelStats
def addStringIfNotEmpty(info, x, sep = ','):
    if x != '':
        if len(info)>0:
            info = info + sep +'  ' + str(x)
        else:
            info = str(x)
    return info
def output_row_to_ML_info_string(df, metric2use):
    info_string0 = str(round(df[metric2use].values[0],2))
    algo = df.Estimator.values[0]
    info_string1 = algo
    ohe = 'OHEau' if df.DoOHEnc.values[0] == 'AU_level' else ''
    info_string1 = addStringIfNotEmpty(info_string1, ohe)
    yt = 'YT' if df.AddYieldTrend.values[0] == 'True' else ''
    info_string1 = addStringIfNotEmpty(info_string1, yt)
    info_string2 = df.Feature_set.values[0]
    dr = df.Data_reduction.values[0] if df.Data_reduction.values[0] != 'none' else ''
    info_string3 = addStringIfNotEmpty('', dr)
    fs = df.Ft_selection.values[0] if df.Ft_selection.values[0] != 'none' else ''
    info_string3 = addStringIfNotEmpty('', fs)
    prct= round(df.Prct_selected_fit.values[0]) if fs != '' else ''
    info_string3 = addStringIfNotEmpty(info_string3, prct, sep=':')
    return [info_string0, info_string1, info_string2, info_string3]
def bars_by_forecast_time(b1, metric2use, mlsettings, var4time, outputDir):
    # in order to assign the same colors and keep a defined order I have to do some workaround
    b1['tmp_est'] = b1['Estimator'].map(lambda x: x if x in mlsettings.benchmarks else 'ML')
    # colors = {'Cat1': "#F28E2B", 'Cat2': "#4E79A7", 'Cat3': "#79706E"}
    colors = {'ML': "#0000FF", 'Null_model': "#969696", 'PeakNDVI': "#FF0000", 'Trend': "#009600"}
    for t in b1[var4time].unique():
        crops = b1['Crop'].unique()
        fig, axs = plt.subplots(ncols=len(crops), figsize=(14, 6))
        ax_c = 0  # ax counter
        # get mas metirc
        ymax = b1[b1[var4time] == t][metric2use].max()
        for crop in crops:
            # in order to assign teh same colors I have to do some workaround
            tmp = b1[(b1[var4time] == t) & (b1['Crop'] == crop)].copy()
            sort_dict = {'Null_model': 0, 'Trend': 1, 'PeakNDVI': 2, 'ML': 3}
            tmp['pltOrder'] = tmp['tmp_est'].map(sort_dict)
            tmp = tmp.sort_values('pltOrder')
            p = sns.barplot(tmp, x="tmp_est", y=metric2use, hue="tmp_est",
                            palette=colors, ax=axs[ax_c], dodge=False, width=0.4, legend="full")
            ml_row = tmp[tmp['tmp_est'] == 'ML']
            [info_string0, info_string1, info_string2, info_string3] = output_row_to_ML_info_string(ml_row, metric2use)
            axs[ax_c].text(0.875, -0.1, metric2use + '= ' + info_string0, transform=axs[ax_c].transAxes,
                           horizontalalignment='center')
            axs[ax_c].text(0.875, -0.14, info_string1, transform=axs[ax_c].transAxes, horizontalalignment='center')
            axs[ax_c].text(0.875, -0.18, info_string2, transform=axs[ax_c].transAxes, horizontalalignment='center')
            axs[ax_c].text(0.875, -0.22, info_string3, transform=axs[ax_c].transAxes, horizontalalignment='center')
            axs[ax_c].get_legend().set_visible(False)
            axs[ax_c].set_title(crop)
            axs[ax_c].set(ylim=(0, ymax * 1.1))
            axs[ax_c].set(xlabel='Model')
            ax_c = ax_c + 1

        h, l = p.get_legend_handles_labels()
        plt.legend(h, l, title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        fig.tight_layout()
        plt.savefig(outputDir + '/' + 'all_model_best1_forecast_time_' + str(t) + '.png')
        plt.close(fig)

def scatter_plots(b1, config, var4time, OutputDir):
    df_regNames = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_REGION_id.csv'))
    crops = b1['Crop'].unique()
    forcTimes = b1[var4time].unique()
    for c in crops:
        for t in forcTimes:
            fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
            axs = axs.flatten()
            df_c_t = b1[(b1['Crop'] == c) & (b1[var4time] == t)].copy()
            sort_dict = {'Null_model': 0, 'Trend': 1, 'PeakNDVI': 2, 'ML': 3}
            df_c_t['pltOrder'] = df_c_t['tmp_est'].map(sort_dict)
            df_c_t = df_c_t.sort_values('pltOrder').reset_index()
            # iterate over pandas df
            for index, row in df_c_t.iterrows():
                # get run_id
                runID = row['runID']
                est = row['Estimator']
                myID = f'{runID:06d}'
                fn_spec = os.path.join(pathlib.Path(config.models_spec_dir), myID + '_' + c + '_' + est + '.json')
                print(fn_spec)
                df = d090_model_wrapper.fit_and_validate_single_model(fn_spec, config, 'tuning' , run2get_mres_only=True)
                lims = [np.floor(np.min([df['yLoo_true'].values, df['yLoo_pred'].values])),
                        np.ceil(np.max([df['yLoo_true'].values, df['yLoo_pred'].values]))]
                r2p = d140_modelStats.r2_nan(df['yLoo_true'].values, df['yLoo_pred'].values)
                for au_code in df['AU_code'].unique():
                    x = df[df['AU_code'] == au_code]['yLoo_true'].values
                    y = df[df['AU_code'] == au_code]['yLoo_pred'].values
                    lbl = df_regNames[df_regNames['AU_code'] == au_code.astype('int')]['AU_name'].values[0]
                    axs[index].scatter(x, y, label=lbl)
                    axs[index].plot(lims, lims, color='black', linewidth=0.5)
                    axs[index].set_title(est + ',R2p=' + str(np.round(r2p, 2)))
                    axs[index].set_xlim(lims)
                    axs[index].set_ylim(lims)
                    axs[index].set_xlabel('Obs')
                    axs[index].set_ylabel('Pred')
                    axs[index].legend(frameon=False, loc='upper left')
            fig.tight_layout()
            plt.savefig(OutputDir + '/' + 'all_model_best1_forecast_time_' + str(t) + '_' + c +'_scatter.png')
            plt.close(fig)

def map_errors(a, b, c):
    print ('pippo')


def accuracy_over_time(mRes, mCountryRes, filename=None):
    # Define dash style
    # there are many AUs, add a columnn to be used with style, 1,2,3,4,5,1,2,3,4,5,1,2..
    # suppress graphics for massive runs
    hideGraph = True
    if (hideGraph):
        plt.ioff()
        plt.switch_backend('agg')  # attempt to get rid of multi thread erros,
        # to have the graph displayed: # plt.ion() # plt.switch_backend('TkAgg') #plt.show()

    dash_base_styles = ["",
                        (4, 1.5),
                        (1, 1),
                        (3, 1, 1, 1, 1, 1),
                        (5, 1, 1, 1)]
    dash_styles = dash_base_styles * 100
    # dash_styles = dash_styles[0:len(mRes)]
    dash_styles = dash_styles[0:len(mRes['AU_code'].unique())]

    # line plot
    fig, axs = plt.subplots(nrows=2, figsize=(12, 6))
    minV, maxV = mRes[['yLoo_true', 'yLoo_pred']].values.min(), mRes[
        ['yLoo_true', 'yLoo_pred']].values.max()
    margin = (maxV - minV) / 20.0
    g = sns.lineplot(x="Year", y="yLoo_true", hue="AU_code", style="AU_code", data=mRes, ax=axs[0], dashes=dash_styles,
                     palette="Spectral", legend='full')
    axs[0].plot(mCountryRes.index, mCountryRes['yLoo_true'], color='green', linewidth=3, linestyle='dashed')
    axs[0].set_ylim(minV - margin, maxV + margin)
    axs[0].axes.set_xlabel('')
    # resize to accomodat legend
    box = g.get_position()
    g.set_position([box.x0, box.y0, box.width * 0.9, box.height])  # resize position
    # Put a legend to the right side
    g.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)

    g = sns.lineplot(x="Year", y="yLoo_pred", hue="AU_code", style="AU_code", data=mRes, ax=axs[1], dashes=dash_styles,
                     palette="Spectral", legend='full')
    axs[1].plot(mCountryRes.index, mCountryRes['yLoo_pred'], color='green', linewidth=3, linestyle='dashed')
    axs[1].set_ylim(minV - margin, maxV + margin)
    # resize to accomodat legend
    box = g.get_position()
    g.set_position([box.x0, box.y0, box.width * 0.9, box.height])  # resize position
    g.legend()._visible = False

    if filename is not None:
        fig.savefig(filename.replace(" ", ""))
    plt.close(fig)

    return

def scatter_plot_accuracy(mRes, title_R2, color_factor, filename=None):
    """
    Scatter plot coloring by a factor
    """
    # suppress graphics for massive runs
    hideGraph = True
    if (hideGraph):
        plt.ioff()
        plt.switch_backend('agg')  # attempt to get rid of multi thread erros,
        # to have the graph displayed: # plt.ion() # plt.switch_backend('TkAgg') #plt.show()

    dash_base_styles = ["",
                        (4, 1.5),
                        (1, 1),
                        (3, 1, 1, 1, 1, 1),
                        (5, 1, 1, 1)]
    dash_styles = dash_base_styles * 100
    # dash_styles = dash_styles[0:len(mRes)]
    dash_styles = dash_styles[0:len(mRes['AU_code'].unique())]

    #scatter plot
    fig = plt.figure(figsize = (6,6))
    base_markers = ('o', 'v', 's', '^', 'X')
    markers = base_markers * 100
    #markers = markers[0:len(mRes)]
    markers = markers[0:len(mRes[color_factor].unique())]
    g = sns.scatterplot(x="yLoo_true", y="yLoo_pred", hue=color_factor, style=color_factor, data=mRes, palette = "Spectral", markers=markers, legend='full')
    sns.regplot(x="yLoo_true", y="yLoo_pred", data=mRes, color='k', scatter_kws={"alpha": 0.0})
    #sns.jointplot(x="yLoo_true", y="yLoo_pred", hue="AU_code", data=mRes)
    minV, maxV = mRes[['yLoo_true', 'yLoo_pred']].values.min(), mRes[['yLoo_true', 'yLoo_pred']].values.max()
    margin = (maxV-minV)/20.0
    plt.ylim(minV-margin, maxV+margin)
    plt.xlim(minV-margin, maxV+margin)
    plt.plot([minV-margin, maxV+margin], [minV-margin, maxV+margin], color= 'k',linewidth=1, linestyle='dashed')
    plt.title('R2_pred='+str(np.round(title_R2,2)))
    box = g.get_position()
    g.set_position([box.x0, box.y0, box.width * 0.85, box.height* 0.85])
    g.legend(loc='lower left', bbox_to_anchor=(1, 0), ncol=1)

    if filename is not None:
        fig.savefig(filename.replace(" ", ""))
    plt.close(fig)


