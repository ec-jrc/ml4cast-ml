import pandas as pd
from lmoments3 import distr
import scipy.stats as stats
from scipy.stats import gamma
import numpy as np
import matplotlib.pyplot as plt

def calc_spi(arr):
    """
    Core calculation of the Standardized Precipitation Index (SPI) on a 1D array.
    First value of the input array is the target value and the rest are historic values.
    Requires at least min 3 valid non-zero historic values to be able to calculate the SPI
    :param ndarray arr: 1D masked array
    :return float : SPI value
    """
    # remove values below 0
    arr.data.clip(0, None, out=arr.data)
    # arr.clip(0, None, out=arr)
    # extract target and hist vals
    target_val = arr[0]
    hist_vals = arr[1:]
    # check the input val
    target_msk = arr.mask[0]
    if np.isnan(target_val) or target_msk:
        return np.nan

    valid_msk = ~hist_vals.mask
    valid = hist_vals[valid_msk]
    # check min 3 valid values, if not set to 0
    if valid_msk.sum() < 3:
        return 0.0

    try:
        data_nonzero_msk = valid != 0
        data_nonzero = valid[data_nonzero_msk]

        # check minimum number of values and if all values are identical
        if len(data_nonzero) < 3 or len(set(data_nonzero)) == 1:
            return 0.0

        fit = distr.gam.lmom_fit(data_nonzero)

        p_zero = (valid == 0).sum(axis=0) / valid_msk.sum(axis=0)

        if target_val >= 0:
            p = distr.gam.cdf(target_val, a=fit['a'], loc=fit['loc'], scale=fit['scale'])
            p = (1 - p_zero) * p + p_zero
        else:
            p = p_zero / 2

    except Exception:
        return 0.0

    p = np.clip(p, 1e-7, 1 - 1e-7)
    spi_out = stats.norm.ppf(p)
    return spi_out

def computeSPIonDF(df, column_name):
    # Compute SPI using gamma function
    # df: dataframe
    # column_name: the column name of teh column containing the precipitation data (same dek over multiple years)
    data =  df[column_name].values
    P0 = data[data == 0].shape[0] / data.shape[0]
    data = data[data != 0]
    # fit with gamma
    fit_dict = distr.gam.lmom_fit(data) # (fit_dict['a'],fit_dict['loc'],fit_dict['scale'])
    # plot
    if False:
        # Extract parameters
        shape = fit_dict['a']
        loc = fit_dict['loc']
        scale = fit_dict['scale']

        # ----------------------------------------------------
        # Generate fitted PDF
        # ----------------------------------------------------
        x = np.linspace(np.min(data), np.max(data), 500)
        pdf = gamma.pdf(x, a=shape, loc=loc, scale=scale)

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------
        plt.figure(figsize=(7, 5))

        plt.hist(data,
                 bins=30,
                 density=True,
                 alpha=0.5,
                 label='Observed data')

        plt.plot(x,
                 pdf,
                 linewidth=2,
                 label='Gamma fit (L-moments)')

        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.show()
    # now compute SPI (dataframe version)
    def SPI(val, fit_dict):
        if val >= 0:
            probVal = distr.gam.cdf(val, a=fit_dict['a'], loc=fit_dict['loc'], scale=fit_dict['scale'])
            probVal = (1 - P0) * probVal + P0
        elif val < 0:
            probVal = P0/2
        if probVal == 0: probVal = 0.0000001
        if probVal == 1: probVal = 0.9999999
        return stats.norm.ppf(probVal, loc=0, scale=1)

    df['SPI_'+column_name] = df[column_name].apply(SPI, fit_dict=fit_dict)
    return df

df = pd.read_csv(r'V:\asap\asap_analysis\chirps v3 vs v2\rain90_v8vsv9.csv')
print(df.head())
# pass it shorte to compare what was done with ENVI/GLIMPSE

df_v8 = df.copy()
tmp = computeSPIonDF(df[df['Y_1989'] < 2019], 'v8')
df_v8['SPI_v8'] = np.nan
df_v8.loc[tmp.index, 'SPI_v8'] = tmp['SPI_v8']


df_v8v9 = computeSPIonDF(df_v8, 'v9')
print(df_v8v9)
histo_ts_v9 = df[df['Y_1989'] < 2024]['v9'].tolist()
print(histo_ts_v9)
spi3 = []
for val in histo_ts_v9:
    tmp = histo_ts_v9.copy()
    tmp.insert(0, val)
    arr = np.array(tmp)
    arr = np.ma.array(arr)
    # add mask if regular array
    if not arr.mask.any():
        arr.mask = np.full(arr.shape, False, dtype=bool)
    spi = calc_spi(arr)
    spi3.append(spi)
    print()
df_v8v9["SPI_v9ope"] = spi3
print(df_v8v9)
print()