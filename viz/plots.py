import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import ternary

def heatmap(data, row_labels, col_labels, ax=None, xpos='bottom', x_axis_label='', figtitle = '',
            show_cbar=True, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if show_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    else:
        cbar = None

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)


    if x_axis_label != '':
        plt.xlabel(x_axis_label, fontsize=10)

    # Decide where the horizontal axes labeling should appear
    if xpos=='bottom':
        ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)
    else:
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)



    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    #ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_aspect('equal')
    if figtitle != '':
        ax.set_title(figtitle, loc='right', fontweight='bold')

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def rgb2hex(x):
    if not isinstance(x, float):
        return matplotlib.colors.to_hex(x)
    else:
        return '#FFFFFF'



def plot_practical_significance(data1, data2, annot2, xlabels1, xlabels2, ylabels2, cmap1, cmap2, rope, alpha,
                                plot_title='', figsize_=(7, 7), filename=""):
    assert data2.shape == annot2.shape

    # FIGURE
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=figsize_, sharey='row',
                                  constrained_layout=True,
                                  gridspec_kw={'height_ratios': [1, len(ylabels2)]})

    # SUBFIG 1
    # Plot SUBFIG 1
    im, _ = heatmap(data1, [''], xlabels1, ax=ax, show_cbar=False, cmap=cmap1, xpos='top', figtitle='rRMSEp')
    annotate_heatmap(im, valfmt="{x:.0f}", size=10, threshold=np.median(data1),
                     textcolors=("white", "black"))

    # TITLE
    fig.suptitle(f'{plot_title}', fontsize=16)

    # SUBFIG 2
    im, cbar = heatmap(data2, ylabels2, xlabels2,
                       ax=ax2, xpos='bottom', x_axis_label='Forecast month',
                       figtitle=f'Practical significance (ROPE={100*rope}% - alpha={alpha})',
                       show_cbar=False, cmap=cmap2)
    for i in range(data2.shape[0]):
        for j in range(data2.shape[1]):
            ax2.text(j, i, f'{annot2.iloc[i, j]}', ha="center", va="center", color='white')
    #plt.tight_layout()
    plt.show()
    if filename != '':
        plt.savefig(filename)


def generate_heatmap_data(scale=5):
    from ternary.helpers import simplex_iterator
    d = dict()
    for (i, j, k) in simplex_iterator(scale):
        d[(i, j, k)] = (i/100, j/100, k/100, 1)
    return d
