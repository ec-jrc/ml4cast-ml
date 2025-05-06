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

    def check_can_compute_trend(x, y, row):
        # count valid in each 3 years period, 4 quartiles
        # we may have less than 12 years, find quartiles
        # Calculate the quartiles
        Q1 = np.percentile(x, 25)
        Q2 = np.percentile(x, 50)
        Q3 = np.percentile(x, 75)
        # Get indexes of elements in each quartile
        Q1_indexes = np.where((x >= x.min()) & (x <= Q1))[0]
        Q2_indexes = np.where((x > Q1) & (x <= Q2))[0]
        Q3_indexes = np.where((x > Q2) & (x <= Q3))[0]
        Q4_indexes = np.where((x > Q3) & (x <= x.max()))[0]
        yBool = np.where(np.isnan(y), 0, 1)
        validByQuartile = [np.sum(yBool[Q1_indexes]), np.sum(yBool[Q2_indexes]), np.sum(yBool[Q3_indexes]),
                           np.sum(yBool[Q4_indexes])]
        # validBy3yrsPeriod = np.where(np.isnan(y), 0, 1).reshape(-1, 3).sum(axis=1)
        # understand which quartiles are important (depending on if we are estimating before or after th 12 years)
        index = np.argmin(np.abs(x - row['Year']))  # 0 if the estimate is on the lft, 11 if right
        # sum valid over the closest two quartiles
        if index == 0:
            sum = np.sum(validByQuartile[:2])
        else:
            sum = np.sum(validByQuartile[-2:])
        # if n >= 4 (one per quartile on avg) and 2 of them are in the closest quartiles, go for trend otherwise use mean
        if np.sum(validByQuartile) >= 4 and sum >= 2:
            return 'yes'
        else:
            return 'no'
    def trend1(row, ny, minYearFeats, yearList):
        """
        This function computes the trend for a given row using ny year before
        """
        yearList = np.array(yearList)
        meanYield = np.nanmean(row[yearList])
        if row['Year'] < minYearFeats:
            # do nothing for years without features
            return np.nan
        else:
            # compute the trend
            years2use = list(range(row['Year']-ny, row['Year']))
            y = row[years2use].values.astype(float)
            x = np.array(years2use).astype(float)
            if check_can_compute_trend(x, y, row) == 'yes':
                ind = ~ np.logical_or(np.isnan(y), np.isnan(x))
                res = stats.theilslopes(y[ind], x[ind])
                est = res[1] + res[0] * row['Year']
                # avoid negative values and truncate the estimate yield to 0.5*mean or 1.5*mean if higher or lower
                if est < 0.25 * meanYield: est = 0.25 * meanYield
                if est > 1.75 * meanYield: est = 1.75 * meanYield
            else:
                est = meanYield
            # ind = ~ np.logical_or(np.isnan(y), np.isnan(x))
            # res = stats.theilslopes(y[ind], x[ind])
            # est = res[1] + res[0] * row['Year']
            # if est < 0: est = 0 #avoid negative yield
            return est

    def trend2(row, ny, minYearFeats, yearList):
        """
        This function computes the trend for a given row using the longest time series (left or right to the year to estimate)
        """
        yearList = np.array(yearList)
        meanYield = np.nanmean(row[yearList])
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
            if check_can_compute_trend(x, y, row) == 'yes':
                ind = ~ np.logical_or(np.isnan(y), np.isnan(x))
                res = stats.theilslopes(y[ind], x[ind])
                est = res[1] + res[0] * row['Year']
                # avoid negative values and truncate the estimate yield to 0.5*mean or 1.5*mean if higher or lower
                if est < 0.25 * meanYield:
                    est = 0.25 * meanYield
                if est > 1.75 * meanYield:
                    est = 1.75 * meanYield
            else:
                est = meanYield
            return est

    yvar = 'Yield' #operate on yield only
    yxDatac['YieldFromTrend'] = np.nan
    # treat each crop / region separately as the stat availability may be different (there should be only one crop here)
    for c in yxDatac['Crop_ID'].unique():
        for r in yxDatac['adm_id'].unique():
            df = yxDatac[(yxDatac['Crop_ID'] == c) & (yxDatac['adm_id'] == r)]
            minYearStats = df.dropna(subset=[yvar])['Year'].min()
            # old: minYearFeats = df.dropna(subset=df.columns[~df.columns.isin(['YieldFromTrend'])].values)['Year'].min()
            minYearFeats = df['YearOfEOS'].min()
            df.loc[:, df['Year'].tolist()] = df[yvar].values  # this is copying yield ts as additional columns
            # use different trend if we have more than ny before the first feature data point
            if minYearFeats - minYearStats > ny:
                # trend estimated with ny years before
                try:
                    df.loc[:, 'YieldFromTrend'] = df.apply(trend1, args=(ny, minYearFeats, df['Year'].tolist()), axis=1)
                except:
                    print('here c1000')
            else:
                # trend estimated using larger time series (left or right)
                # a = trend2(df.iloc[0], ny, minYearFeats, df['Year'].tolist())
                # df.loc[:, 'YieldFromTrend'] = df.apply(trend2, args=(ny, minYearFeats, df['Year'].astype(str).tolist()), axis=1)
                df.loc[:, 'YieldFromTrend'] = df.apply(trend2, args=(ny, minYearFeats, df['Year'].tolist()), axis=1)
            # add the trend to yxDatac
            yxDatac.loc[(yxDatac['Crop_ID'] == c) & (yxDatac['adm_id'] == r), 'YieldFromTrend'] = df['YieldFromTrend']
    return yxDatac