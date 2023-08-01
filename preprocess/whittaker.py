#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WHITTAKER-EILERS SMOOTHER in Python 3 using numpy and scipy
based on the work by Eilers [1].
    [1] P. H. C. Eilers, "A perfect smoother",
        Anal. Chem. 2003, (75), 3631-3636
coded by M. H. V. Werts (CNRS, France)
tested on Anaconda 64-bit (Python 3.6.4, numpy 1.14.0, scipy 1.0.0)
Read the license text at the end of this file before using this software.
Warm thanks go to Simon Bordeyne who pioneered a first (non-sparse) version
of the smoother in Python.
"""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import splu


def speyediff(N, d, format='csc'):
    """
    (utility function)
    Construct a d-th order sparse difference matrix based on
    an initial N x N identity matrix

    Final matrix (N-d) x N
    """

    assert not (d < 0), "d must be non negative"
    shape = (N - d, N)
    diagonals = np.zeros(2 * d + 1)
    diagonals[d] = 1.
    for i in range(d):
        diff = diagonals[:-1] - diagonals[1:]
        diagonals = diff
    offsets = np.arange(d + 1)
    spmat = sparse.diags(diagonals, offsets, shape, format=format)
    return spmat


def whittaker_smooth_m(y0, lmbd, no_data_val, d=2):
    """
    Implementation of the Whittaker smoothing algorithm,
    based on the work by Eilers [1].
    [1] P. H. C. Eilers, "A perfect smoother", Anal. Chem. 2003, (75), 3631-3636

    The larger 'lmbd', the smoother the data.
    For smoothing of a complete data series, sampled at equal intervals
    This implementation uses sparse matrices enabling high-speed processing
    of large input vectors

    ---------

    Arguments :

    y       : vector containing raw data
    lmbd    : parameter for the smoothing algorithm (roughness penalty)
    d       : order of the smoothing

    ---------
    Returns :

    z       : vector of the smoothed data.
    """

    # avoid modifing input
    y = np.copy(y0)

    if no_data_val is None:
        w = np.where(np.isnan(y), 0, 1)
        y[np.where(np.isnan(y))] = 0
    else:
        w = np.where(y == no_data_val, 0, 1)
    m = len(y)

    W = sparse.diags(w, 0, shape=(m, m), format="csc")

    # E = sparse.eye(m, format='csc')
    D = speyediff(m, d, format='csc')
    coefmat = W + lmbd * D.conj().T.dot(D)
    z = splu(coefmat).solve(y)
    return z

def whittaker_smooth_m_withWeights(y0, lmbd, w, no_data_val, d=2):
    """
    Implementation of the Whittaker smoothing algorithm,
    based on the work by Eilers [1].
    [1] P. H. C. Eilers, "A perfect smoother", Anal. Chem. 2003, (75), 3631-3636

    The larger 'lmbd', the smoother the data.
    For smoothing of a complete data series, sampled at equal intervals
    This implementation uses sparse matrices enabling high-speed processing
    of large input vectors

    ---------

    Arguments :

    y       : vector containing raw data
    lmbd    : parameter for the smoothing algorithm (roughness penalty)
    w       : weight vector
    d       : order of the smoothing

    ---------
    Returns :

    z       : vector of the smoothed data.
    """

    # avoid modifing input
    y = np.copy(y0)

    if no_data_val is None:
        w0 = np.where(np.isnan(y), 0, 1)
        y[np.where(np.isnan(y))] = 0
    else:
        w0 = np.where(y == no_data_val, 0, 1)
        y[np.where(y == no_data_val)] = 0
    w = np.multiply(w0,w)
    m = len(y)

    W = sparse.diags(w, 0, shape=(m, m), format="csc")

    # E = sparse.eye(m, format='csc')
    D = speyediff(m, d, format='csc')
    coefmat = W + lmbd * D.conj().T.dot(D)
    z = splu(coefmat).solve(w*y)
    return z