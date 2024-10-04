import numpy as np
from scipy import stats

def add_yield_trend_estimate(yxDatac, ny):
    """
    Add a feature YFromTrend (the y value estimated by the y time trend of a given admin unit).
    The type of trend computation is different depending on how long is the time series of stats data we have.
    if we have at least ny years of stat before timeRange[0] we use ny before year YYYY to estimate yield of YYYY
    else (less than ny years) we use Franz's idea:
    to avoid using info before AND after (not correct because not possible in NRT), we assume that there is no difference between forward and backward y trend estimates,
    meaning that y of year YYYY can be estimated from the time series YYYY-n : YYYY-1 or from the one YYYY+1 : YYYY+n
    So, we locate the year YYYY in the time series. It will have nb year before and na year after.
    We use the years after (if na>nb) or before (if nb>na) to estimate the trend and compute the yield of YYYY.
    Note: computing the y trend estimate beforehand (not in the double loop) has the drawback that when holding out year YYYY (e.g. 2005),
    the model will be tuned with a trend feature computed using YYYY for the years after YYYY (e.g. 2006, 2007, ..).
    This could be avoided computing the trend in the loop and thus excluding YYYY of the outer loop and YYYY2 of inner loop.
    However: 1) with scikit CV for hyperpar optimisation we do not have access to inner loop data, and 2) although not using YYYY,
    we would often use data before and after YYYY (e.g. when YYYY is 2005 and we are estimating 2009).
    Therefore, we opt for some form of info leakage that is due to precomputing Y trend of YYYY. Nevertheless, as in operation,
    data after YYYY are never used to estimate its y value from the trend.
    """
    # ny is number of years of y data before the first year with features AND number of years for trend computation


    def trend1(row, ny, minYearFeats):
        """
        This function computes the trend for a given row using ny year before
        """
        if row['Year'] < minYearFeats:
            # do nothing for years without features
            return np.nan
        else:
            # compute the trend
            years2use = list(range(row['Year']-ny, row['Year']))
            y = row[years2use].values.astype(float)
            x = np.array(years2use).astype(float)
            ind = ~ np.logical_or(np.isnan(y), np.isnan(x))
            res = stats.theilslopes(y[ind], x[ind])
            return res[1] + res[0] * row['Year']

    def trend2(row, ny, minYearFeats, yearList):
        """
        This function computes the trend for a given row using the longest time series (left or right to the year to estimate)
        """
        yearList = np.array(yearList)
        if row['Year'] < minYearFeats:
            # do nothing for years without features
            return np.nan
        else:
            # find the largest arm of the time series (before or after)
            leftYears = yearList[yearList < row['Year']]
            rightYears = yearList[yearList > row['Year']]
            if len(leftYears) >= len(rightYears):
                #use the left arm
                years2use = leftYears
                if len(years2use) > ny:
                    #keep only ny, the last ones
                    years2use = years2use[-ny:]
            else:
                #use the right arm
                years2use = rightYears
                if len(years2use) > ny:
                    #keep only ny, the first ones
                    years2use = years2use[0:ny]
            y = row[years2use].values.astype(float)
            x = np.array(years2use).astype(float)
            ind = ~ np.logical_or(np.isnan(y), np.isnan(x))
            res = stats.theilslopes(y[ind], x[ind])
            return res[1] + res[0] * row['Year']

    yvar = 'Yield' #operate on yield only
    yxDatac['YieldFromTrend'] = np.nan
    # treat each crop / region separately as the stat availability may be different (there should be only one crop here)
    for c in yxDatac['Crop_ID'].unique():
        for r in yxDatac['adm_id'].unique():
            df = yxDatac[(yxDatac['Crop_ID'] == c) & (yxDatac['adm_id'] == r)]
            minYearStats = df.dropna(subset=[yvar])['Year'].min()
            minYearFeats = df.dropna(subset=df.columns[~df.columns.isin(['YieldFromTrend'])].values)['Year'].min()
            #add year columns
            newColumns = list(map(str, df['Year'].tolist()))
            # df.loc[:,newColumns] = df[yvar].values
            # df[newColumns] = df[yvar].values
            df.loc[:, df['Year'].tolist()] = df[yvar].values
            # use different trend if we have more than ny before the first feature data point
            if minYearFeats - minYearStats > ny:
                # trend estimated with ny years before
                df.loc[:, 'YieldFromTrend'] = df.apply(trend1, args=(ny, minYearFeats), axis=1)
            else:
                # trend estimated using larger time series (left or right)
                df.loc[:, 'YieldFromTrend'] = df.apply(trend2, args=(ny, minYearFeats, df['Year'].tolist()), axis=1)
            # add the trend to yxDatac
            yxDatac.loc[(yxDatac['Crop_ID'] == c) & (yxDatac['adm_id'] == r), 'YieldFromTrend'] = df['YieldFromTrend']
    return yxDatac