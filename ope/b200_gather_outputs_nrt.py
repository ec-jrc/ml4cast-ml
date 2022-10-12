import os, glob
import numpy as np
import pandas as pd

import src.constants as cst
import ml.modeller as modeller
import b05_Init


def combine_models_consistency_check(fns):
    # compare best mode, lasso and peak and adjust extreme predictions, if any
    # Avoid very low estimates
    # if y_prct of best is <= 0.1 and lasso have a larger one, take that of lasso and put in best
    # if y_prct of new best (can be best or lasso) is still <= 0.1 and peak have a larger one, take that of lasso and put in best
    # Avoid very high estimates
    # same on 0.9 percentile
    # check if still y_percent < 0.1 and forecast < min yield) --> min  Yield or  > 0.9 -- > max Yield
    # in that case use min or max observed yield as forecasted yield

    if len([x for x in fns if 'Lasso' not in x and 'PeakNDVI' not in x]) > 0: #if len(fns) == 3:
        fn_1 = [x for x in fns if 'Lasso' not in x and 'PeakNDVI' not in x][0]
        df_best = pd.read_csv(fn_1, index_col=0)

        fn_2 = [x for x in fns if 'Lasso' in x][0]
        df_lasso = pd.read_csv(fn_2, index_col=0)

        fn_3 = [x for x in fns if 'PeakNDVI' in x][0]
        df_peak = pd.read_csv(fn_3, index_col=0)

        # Avoid very low estimates
        # if y_prct of best is <= 0.1 and lasso have a larger one, take that of lasso and put in best
        df_best.loc[(df_best.fyield_percentile <= 0.1) & (df_best.fyield_percentile < df_lasso.fyield_percentile), :] =\
            df_lasso.loc[(df_best.fyield_percentile <= 0.1) & (df_best.fyield_percentile < df_lasso.fyield_percentile), :]
        # if y_prct of new best (can be best or lasso) is still <= 0.1 and peak have a larger one, take that of lasso and put in best
        df_best.loc[(df_best.fyield_percentile <= 0.1) & (df_best.fyield_percentile < df_peak.fyield_percentile), :] = \
            df_peak.loc[(df_best.fyield_percentile <= 0.1) & (df_best.fyield_percentile < df_peak.fyield_percentile), :]
        # Avoid very high estimates
        # same on 0.9 percentile
        df_best.loc[(df_best.fyield_percentile >= 0.9) & (df_best.fyield_percentile > df_lasso.fyield_percentile), :] = \
            df_lasso.loc[(df_best.fyield_percentile >= 0.9) & (df_best.fyield_percentile > df_lasso.fyield_percentile), :]
        df_best.loc[(df_best.fyield_percentile >= 0.9) & (df_best.fyield_percentile > df_peak.fyield_percentile), :] = \
            df_peak.loc[(df_best.fyield_percentile >= 0.9) & (df_best.fyield_percentile > df_peak.fyield_percentile), :]

    elif len([x for x in fns if 'Lasso' not in x and 'PeakNDVI' not in x]) == 0: #len(fns) == 2:
        # same if lasso was already best
        fn_1 = [x for x in fns if 'Lasso' in x][0]
        df_best = pd.read_csv(fn_1, index_col=0)

        fn_2 = [x for x in fns if 'PeakNDVI' in x][0]
        df_lasso = pd.read_csv(fn_2, index_col=0)
        df_best.loc[(df_best.fyield_percentile <= 0.1) & (df_best.fyield_percentile < df_lasso.fyield_percentile), :] = \
            df_lasso.loc[(df_best.fyield_percentile <= 0.1) & (df_best.fyield_percentile < df_lasso.fyield_percentile), :]
        df_best.loc[(df_best.fyield_percentile >= 0.9) & (df_best.fyield_percentile > df_lasso.fyield_percentile), :] = \
            df_lasso.loc[(df_best.fyield_percentile >= 0.9) & (df_best.fyield_percentile > df_lasso.fyield_percentile), :]
    else:
        print('Inconsistent number of files. End of the world. Stopping now')
        AssertionError
    # check if still y_percent < 0.1 and forecast < min yield) --> min  Yield or  > 0.9 -- > max Yield
    if sum(df_best.fyield_percentile <= 0.1) > 0:
        # min
        select_rows_min = (df_best.fyield < df_best.min_obs_yield) & (df_best.fyield_percentile <= 0.1)
        df_best.loc[select_rows_min, 'fyield'] = df_best.loc[select_rows_min, 'min_obs_yield']
        # update variables related to fyield
        df_best.loc[select_rows_min, ['fyield_percentile']] = 0
        df_best.loc[select_rows_min, ['algorithm']] = 'min'
        df_best.loc[select_rows_min, ['fyield_SD_Bootstrap_1yr', 'cv_mae']] = np.nan

        df_best.loc[select_rows_min, 'yield_diff_pct'] = 100 * (df_best.loc[select_rows_min, 'fyield'] -
                                                             df_best.loc[select_rows_min, 'avg_obs_yield']) / \
                                                      df_best.loc[select_rows_min, 'avg_obs_yield']
        df_best.loc[select_rows_min, 'fproduction(fyield*avg_obs_area)'] = \
            df_best.loc[select_rows_min, 'fyield'] * df_best.loc[select_rows_min, 'avg_obs_area']
        df_best.loc[select_rows_min, 'fproduction_percentile'] = np.nan

        # max
        select_rows_max = (df_best.fyield > df_best.max_obs_yield) & (df_best.fyield_percentile >= 0.9)
        df_best.loc[select_rows_max, 'fyield'] = df_best.loc[select_rows_max, 'max_obs_yield']
        # update variables related to fyield
        df_best.loc[select_rows_max, ['fyield_percentile']] = 0
        df_best.loc[select_rows_max, ['algorithm']] = 'max'
        df_best.loc[select_rows_max, ['fyield_SD_Bootstrap_1yr', 'cv_mae']] = np.nan

        df_best.loc[select_rows_max, 'yield_diff_pct'] = 100 * (df_best.loc[select_rows_max, 'fyield'] -
                                                             df_best.loc[select_rows_max, 'avg_obs_yield']) / \
                                                      df_best.loc[select_rows_max, 'avg_obs_yield']
        df_best.loc[select_rows_max, 'fproduction(fyield*avg_obs_area)'] = \
            df_best.loc[select_rows_max, 'fyield'] * df_best.loc[select_rows_max, 'avg_obs_area']
        df_best.loc[select_rows_max, 'fproduction_percentile'] = np.nan

    return df_best


def main(target, folder, filename=''):
    """
    Gather crop-specific forecasts and generate a nation-scale forecast
    :param target: Area of interest
    :param dir: directory where crop-specific forecasts are stored
    :param filename: output filename. If none provided a default
    """
    tgt_dir = os.path.join(cst.odir, target)
    fns = [x for x in glob.glob(os.path.join(folder, '*')) if 'national' not in x and 'consolidated' not in x]

    df_stats = pd.read_pickle(os.path.join(tgt_dir, f'{target}_stats.pkl'))
    crop_list, production, ppercentile, yields, yieldsdiff = [], [], [], [], []
    project = b05_Init.init(target)
    crop_Names = list(project['crop_names'].values())
    #for crop_name in ['Barley', 'Durum wheat', 'Soft wheat']:
    for crop_name in crop_Names:
        print(crop_name)
        fns_crop = [x for x in fns if crop_name in x]
        df_f = combine_models_consistency_check(fns_crop)
        fn_out = '_'.join(fns_crop[0].split('_')[0:-1] + ['consolidated.csv'])

        # Subset stats
        df_stats_i = df_stats[df_stats['Crop_name'] == crop_name]
        df_stats_i = df_stats_i[df_stats_i['AU_name'].isin(df_f['Region_name'])]

        # Recompute production percentile
        for region in df_f['Region_name']:
            if df_f.loc[df_f['Region_name'] == region, 'fproduction_percentile'].isna().sum() == 1:
                stats_region = df_stats_i[df_stats_i['AU_name'] == region]
                df_f.loc[df_f['Region_name'] == region, 'fproduction_percentile'] = \
                    modeller.percentile_below(stats_region['Production'],
                                              df_f.loc[df_f['Region_name'] == region, 'fproduction(fyield*avg_obs_area)'].values)
        # Save updated values
        df_f.to_csv(fn_out)
        # compute national stats
        prod = df_f['fproduction(fyield*avg_obs_area)'].sum()
        area = df_f['avg_obs_area'].sum()

        # Define a lambda function to compute the weighted mean:
        wm = lambda x: np.average(x, weights=df_stats_i.loc[x.index, "Area"])

        df_stats_sum = df_stats_i.groupby(['Year']).agg(
            Production=pd.NamedAgg(column="Production", aggfunc="sum"),
            nat_yield=pd.NamedAgg(column="Yield", aggfunc=wm)
        )

        crop_list.append(crop_name)
        production.append(prod)
        yields.append(prod / area)
        yieldsdiff.append(100 * ((prod / area) - df_stats_sum['nat_yield'].mean()) / df_stats_sum['nat_yield'].mean())
        ppercentile.append(modeller.percentile_below(df_stats_sum['Production'], prod))

    df = pd.DataFrame({'Crop_Name': crop_list,
                       'fyield': yields,
                       'yield_diff_pct': yieldsdiff,
                       'fproduction(fyield*avg_obs_area)': production,
                       'fproduction_percentile': ppercentile})

    if filename == '':
        fn_parts = os.path.basename(fn_out).split('_')
        fn_parts[3] = 'national'
        filename = os.path.join(folder, '_'.join(fn_parts))
    df.to_csv(filename)


if __name__ == '__main__':
    main(target='Algeria', folder=cst.folder_b200_gather_outputs_nrt)
