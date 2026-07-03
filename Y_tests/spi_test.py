
import numpy as np
import pandas as pd
from lmoments3 import distr
import scipy.stats as stat



def computeSPIonDF(df, column_name):
    # Compute SPI using gamma function
    # df: dataframe
    # column_name: the column name of teh column containing the precipitation data (same dek over multiple years)
    data =  df[column_name].values
    P0 = data[data == 0].shape[0] / data.shape[0]
    data = data[data != 0]
    # fit with gamma
    fit_dict = distr.gam.lmom_fit(data) # (fit_dict['a'],fit_dict['loc'],fit_dict['scale'])
    # now compute SPI (dataframe version)
    def SPI(val, fit_dict):
        if val >= 0:
            probVal = distr.gam.cdf(val, a=fit_dict['a'], loc=fit_dict['loc'], scale=fit_dict['scale'])
            probVal = (1 - P0) * probVal + P0
        elif val < 0:
            probVal = P0/2
        if probVal == 0: probVal = 0.0000001
        if probVal == 1: probVal = 0.9999999
        return stat.norm.ppf(probVal, loc=0, scale=1)

    df['SPI_'+column_name] = df[column_name].apply(SPI, fit_dict=fit_dict)
    return df


# TEST rainfall use to test, in Ronco's we will use directly "data" the wighted precipitation avg, all yearly records of dekad d, d in (1, 36)
rainfall_data = pd.read_csv(r'X:\PY\SPI\data\monthly_data.csv')
# add the month to the date
rainfall_data['date'] = pd.to_datetime(rainfall_data['date'])
rainfall_data['month'] = rainfall_data['date'].dt.month
# get all yeraly data for a given month, say month 1
data_m = rainfall_data[rainfall_data['month']==1].copy()
df_with_SPI = computeSPIonDF(data_m, 'TotalPrecipitation')
print(df_with_SPI)
#TEST END