from sklearn.mixture import GaussianMixture
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


def gam(X, nc, au, date, bins = None, aic_threshold = 0):
    """
    Add gaussian mixture, test nc components
    Note: as we have AFI, we cannot use the NDVI data directly but need to reconstruct data from the histogram
    Parameters:
    :param X: x values (or the histogram, see bins), in our case NDVI values
    :param nc: maximum number of gaussian components to be tested
    :param bins: if provided it means that X is a histogram of NDVI counts per bib
    # https://www.astroml.org/book_figures/chapter4/fig_GMM_1D.html
    """
    # fit models with 1-10 components
    if nc>3:
        print('Gamm: no more than 3 components handles')
        exit()
    N = np.arange(1, nc+1)
    models = [None for i in range(len(N))]
    if bins is not None:
        # reconstruct data from histogram and bins
        # get bins mid-point
        binsize = (bins[1]-bins[0])
        midpoints = bins[0:-1] + binsize/2.0
        # replicate midpoints the number of times prescribed bu the histo
        X = np.repeat(midpoints, np.round(X,0).astype('int'))
        nbins = len(bins) - 1
    else:
        nbins = 100

    X = X.reshape(-1, 1)
    for i in range(len(N)):
        models[i] = GaussianMixture(N[i]).fit(X)
    # test repetition and Silohuette coefficient (failed: teh score cany be evaluate with 1 class)
    # modelsr = [None for i in range(len(N))]
    # def SelBest(arr: list, X: int) -> list:
    #     '''
    #     returns the set of X configurations with shorter distance
    #     '''
    #     dx = np.argsort(arr)[:X]
    #     return arr[dx]
    # iterations = 20
    # sils = []
    # sils_err = []
    # for i in range(len(N)):
    #     tmp_sil = []
    #     for _ in range(iterations):
    #         modelsr[i] = GaussianMixture(N[i], n_init=4).fit(X)
    #         labels = modelsr[i].predict(X)
    #         sil = metrics.silhouette_score(X, labels, metric='euclidean')
    #         tmp_sil.append(sil)
    #     val = np.mean(SelBest(np.array(tmp_sil), int(iterations / 5)))
    #     err = np.std(tmp_sil)
    #     sils.append(val)
    #     sils_err.append(err)



    # compute the AIC and the BIC
    AIC = [m.aic(X) for m in models]
    BIC = [m.bic(X) for m in models]


    # ------------------------------------------------------------
    # Plot the results
    #  We'll use three panels:
    #   1) data + best-fit mixture
    #   2) AIC and BIC vs number of components
    #   3) probability that a point came from each component

    fig = plt.figure(figsize=(12, 3))
    #fig.subplots_adjust(left=0.12, right=0.97,
    #                    bottom=0.21, top=0.9, wspace=0.5)
    fig.subplots_adjust(left=0.08, right=0.97,
                        bottom=0.16, top=0.86, wspace=0.5)

    # plot 1: data + best-fit mixture
    ax = fig.add_subplot(141)
    if aic_threshold == 0:
        M_best = models[np.argmin(AIC)]
    else:
        #compute difference between model with one component
        AIC_diff = AIC-AIC[0]
        AIC_t = np.array(AIC)
        AIC_t[AIC_diff>aic_threshold] = np.inf
        M_best = models[np.argmin(AIC_t)]

    # make and index to reorder components in descending order of the mean (i.e. class 1 mean > class 2, ..)
    ordr = np.flip(np.argsort(M_best.means_[:,0]))
    NC_best = len(M_best.means_[:,0])

    # use bins midpoints to compute pdf, so that P(bin@midpoint) = pdf(midpoint) * binsize
    x = midpoints

    # Compute the log-likelihood of each sample.
    logprob = M_best.score_samples(x.reshape(-1, 1))
    # Evaluate the components' density for each sample.
    responsibilities = M_best.predict_proba(x.reshape(-1, 1))
    responsibilities = responsibilities[:,ordr]
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    ax.hist(X, bins=bins, density=True, histtype='stepfilled', alpha=0.4)
    ax.plot(x, pdf, '-k')
    col = ['red','blue','yellow']
    for i in range(NC_best):
        ax.plot(x, pdf_individual[:,i], c=col[i])
    ax.text(0.04, 0.96, "Best-fit Mixture",
            ha='left', va='top', transform=ax.transAxes)
    ax.set_xlabel('$NDVI$')
    ax.set_ylabel('$p(NDVI)$')

    # plot 2: AIC and BIC
    ax = fig.add_subplot(142)
    ax.plot(N, AIC, '-k', label='AIC')
    ax.plot(N, BIC, '--k', label='BIC')
    ax.set_xlabel('n. components')
    ax.set_ylabel('information criterion')
    ax.legend(loc=2)

    # Plot P of each original bins
    ax = fig.add_subplot(143)
    for i in range(NC_best):
        ax.plot(x, responsibilities[:, i], c=col[i])
    ax.set_xlabel('$NDVI$')
    ax.set_ylabel(r'$p({\rm class}|x)$')

    # plot 3: posterior probabilities for each component
    ax = fig.add_subplot(144)

    p = responsibilities
    #p = p[:, (1, 0, 2)]  # rearrange order so the plot looks better
    p = p.cumsum(1).T

    ax.fill_between(x, 0, p[0], color=col[0], alpha=0.3)
    ax.text(x[np.argmax(responsibilities[:,0])], 0.3, 'class 1', rotation='vertical')
    if p.shape[0] == 2:
        ax.fill_between(x, p[0], p[1], color=col[1], alpha=0.3)
        ax.text(x[np.argmax(responsibilities[:,1])], 0.5, 'class 2', rotation='vertical')
    if p.shape[0] == 3:
        ax.fill_between(x, p[1], 1, color=col[2], alpha=0.3)
        ax.text(x[np.argmax(responsibilities[:,2])], 0.3, 'class 3', rotation='vertical')
   # ax.set_xlim(0 , 1)
    #ax.set_ylim(0, 1)
    ax.set_xlabel('$NDVI$')
    ax.set_ylabel(r'$p({\rm class}|x)$')
    strTtl = au + ' ' + str(date)+ ', Means = ' + ', '.join([str(np.round(x,2)) for x in M_best.means_[ordr,0].tolist()]) + \
        '; P(C1), P(C2) = ' + ', '.join([str(np.round(x,2)) for x in M_best.weights_[ordr].tolist()])
    if aic_threshold !=0:
        strTtl = strTtl + ', AIC thrshld = ' + str(aic_threshold)

    fig.suptitle(strTtl)
    # the first resp is the one with the highest mean
    return fig, responsibilities





