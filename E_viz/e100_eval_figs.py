import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd


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


