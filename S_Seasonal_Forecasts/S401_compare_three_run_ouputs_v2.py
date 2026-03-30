import pandas as pd
from A_config import a10_config
import os
import sys
import glob
import calendar
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skill_metrics as sm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from D_modelling import d140_modelStats
from S_Seasonal_Forecasts import S501_compare_season_obs_vs_SF_v2
from scipy.stats import shapiro, ttest_rel
from F_post_processsing import F100_analyze_hindcast_output
from compactletterdisplay.pairwise_comp import anova_cld
"""
Compare two results (e.g. without SF vs with SF) 
"""

# configuration files and run names
# baseDir = r'V:\foodsec\Projects\SNYF\SIDv\TN\SF'
# cf1 = os.path.join(baseDir, 'NO_SF_baseline\TNMultiple_WC-Tunisia-ASAP_config.json')
# rn1 = 'TNv_NoSF'
# short_name1 = 'noSF'
#
# cf2 = os.path.join(baseDir, 'ObsAsSF\TNMultiple_WC-Tunisia-ASAP_config_ObsAsForecast.json')
# rn2 = 'TNv_ObsAsSF'
# short_name2 = 'ObsAsSF'
#
# cf3 = os.path.join(baseDir, 'SF\TNMultiple_WC-Tunisia-ASAP_config_SfAsForecast.json')
# rn3 = 'TNv_SfAsSF'
# short_name3 = 'SF'

# baseDir = r'V:\foodsec\Projects\SNYF\SIDvs\ZA\summer2025data\SF'
# cf1 = os.path.join(baseDir, 'SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config1235.json')
# rn1 = 'ZA_NoSF'
# short_name1 = 'noSF'
#
# cf2 = os.path.join(baseDir, 'SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config1235_ObsAsForecast.json')
# rn2 = 'ZA_ObsAsSF'
# short_name2 = 'ObsAsSF'
#
# cf3 = os.path.join(baseDir, 'SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config1235_SfAsForecast.json')
# rn3 = 'ZA_SfAsSF'
# short_name3 = 'SF'

# SF2
baseDir = r'V:\foodsec\Projects\SNYF\SIDvs\ZA\summer2025data\SF2'
cf1 = os.path.join(baseDir, 'SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config12345.json')
rn1 = 'ZA_NoSF'
short_name1 = 'noSF'

cf2 = os.path.join(baseDir, 'SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config12345_ObsAsForecast.json')
rn2 = 'ZA_ObsAsSF'
short_name2 = 'ObsAsSF'

cf3 = os.path.join(baseDir, 'SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config12345_SfAsForecast.json')
rn3 = 'ZA_SfAsSF'
short_name3 = 'SF'

# # SF3
# baseDir = r'V:\foodsec\Projects\SNYF\SIDvs\ZA\summer2025data\SF3au'
# cf1 = os.path.join(baseDir, 'SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config12345.json')
# rn1 = 'ZA_NoSF'
# short_name1 = 'noSF'
#
# cf2 = os.path.join(baseDir, 'SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config12345_ObsAsForecast.json')
# rn2 = 'ZA_ObsAsSF'
# short_name2 = 'ObsAsSF'
#
# cf3 = os.path.join(baseDir, 'SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config12345_SfAsForecast.json')
# rn3 = 'ZA_SfAsSF'
# short_name3 = 'SF'

# plot Tab results or not
plotTab = False
# plot Peak
plotPeak = False
# Plot Taylor
makeTaylor = False
# Wilcoxon level
alphaW = 0.05
########################################################
def squared_error_from_mRes_file(fn):
    mRes = pd.read_csv(fn)
    y_pred = mRes["yLoo_pred"]
    y_true = mRes["yLoo_true"]
    # Mask where BOTH are not NaN
    mask = y_pred.notna() & y_true.notna()
    # Keep only paired values
    y_pred_valid = y_pred[mask]
    y_true_valid = y_true[mask]
    # Compute squared error
    err2 = (y_true_valid - y_pred_valid) ** 2
    # Convert to numpy array
    return err2.to_numpy()

def compute_paired_test(group, runs, configs, short_names):
    # ['noSF', 'ObsAsSF', 'SF']
    out = group.copy()
    out['wilcoxon'] = np.nan
    # get ref mres (ML_obs)
    shortName = "noSF"
    runID = group[(group['run_short_name'] == shortName) & (group['Estimator']!='PeakFPAR')]['runID'].iloc[0]
    myID = f'ID_{runID:06d}'
    idx = short_names.index(shortName)
    dir = configs[idx].models_out_dir
    pattern = os.path.join(dir, f"{myID}*_mres.csv")
    mRes_file = glob.glob(pattern)
    if len(mRes_file) > 1:
        print('compute_paired_test, more than one file')
        sys.exit()
    err2_noSF = squared_error_from_mRes_file(mRes_file[0])

    # get Ml_ObsAsSF
    shortName = "ObsAsSF"
    runID = group[(group['run_short_name'] == shortName) & (group['Estimator'] != 'PeakFPAR')]['runID'].iloc[0]
    myID = f'ID_{runID:06d}'
    idx = short_names.index(shortName)
    dir = configs[idx].models_out_dir
    pattern = os.path.join(dir, f"{myID}*_mres.csv")
    mRes_file = glob.glob(pattern)
    if len(mRes_file) > 1:
        print('compute_paired_test, more than one file')
        sys.exit()
    err2_ObsAsSF = squared_error_from_mRes_file(mRes_file[0])
    # test normality for paired t-test
    stat, p = shapiro(err2_noSF - err2_ObsAsSF)
    if p > alphaW:
        print('SUBNAT D is not normal')
    test = ttest_rel(err2_noSF, err2_ObsAsSF, alternative="greater")
    # # do wilcoxon 1 (noSF vs ObsAsSF)
    # test = d140_modelStats.paired_wilcoxon(err2_noSF, err2_ObsAsSF, alternative="greater", alpha=alphaW, verbose=False)
    out.loc[(out['run_short_name'] == shortName) & (out['Estimator'] != 'PeakFPAR'), 'wilcoxon'] = test[1]

    # get ML_sf
    shortName = "SF"
    runID = group[(group['run_short_name'] == shortName) & (group['Estimator'] != 'PeakFPAR')]['runID'].iloc[0]
    myID = f'ID_{runID:06d}'
    idx = short_names.index(shortName)
    dir = configs[idx].models_out_dir
    pattern = os.path.join(dir, f"{myID}*_mres.csv")
    mRes_file = glob.glob(pattern)
    if len(mRes_file) > 1:
        print('compute_paired_test, more than one file')
        sys.exit()
    err2_SF = squared_error_from_mRes_file(mRes_file[0])
    stat, p = shapiro(err2_noSF - err2_SF)
    if p > alphaW:
        print('SUBNAT D is not normal')
    test = ttest_rel(err2_noSF, err2_SF, alternative="greater")
    # # do wilcoxon 2 (noSF vs SF)
    # test = d140_modelStats.paired_wilcoxon(err2_noSF, err2_SF, alternative="greater", alpha=alphaW, verbose=False)
    out.loc[(out['run_short_name'] == shortName) & (out['Estimator'] != 'PeakFPAR'), 'wilcoxon'] = test[1]

    # bins = np.histogram_bin_edges(err2_noSF, bins=30)
    # plt.figure(figsize=(7, 5))
    # # err2_SF with same bins
    # plt.hist(err2_ObsAsSF, bins=bins, alpha=0.6, color='blue', label='err2_ObsAsSF', edgecolor='black')
    # # err2_noSF with same bins
    # plt.hist(err2_noSF, bins=bins, alpha=0.4, color='orange', label='err2_noSF', edgecolor='black')
    # plt.xlabel("Squared Error")
    # plt.ylabel("Frequency")
    # plt.title("Distribution of Squared Errors")
    # plt.legend()
    # plt.show()
    return out
def compute_paired_test_nat(group):
    # ['noSF', 'ObsAsSF', 'SF']
    out = group.copy()
    out['wilcoxon'] = np.nan
    # ['ML_noSF', 'ML_ObsAsSF', 'ML_SF']
    # make squere error of ML_noSF
    err2_noSF = (out[out['model_name']=='ML_noSF']['y_true'].iloc[0].to_numpy() -
                 out[out['model_name']=='ML_noSF']['y_pred'].iloc[0].to_numpy())**2

    # now Ml_ObsAsSF
    err2_ObsAsSF = (out[out['model_name']=='ML_ObsAsSF']['y_true'].iloc[0].to_numpy() -
                    out[out['model_name']=='ML_ObsAsSF']['y_pred'].iloc[0].to_numpy())**2
    # test normality for paired t-test
    stat, p = shapiro(err2_noSF - err2_ObsAsSF)
    if p > alphaW:
        print('Nat D is not normal')
    test = ttest_rel(err2_noSF, err2_ObsAsSF, alternative="greater")
    # # do wilcoxon 1 (noSF vs ObsAsSF)
    # test = d140_modelStats.paired_wilcoxon(err2_noSF, err2_ObsAsSF, alternative="greater", alpha=alphaW, verbose=False)
    out.loc[(out['model_name'] == 'ML_ObsAsSF'), 'wilcoxon'] = test[1]

    # now ML_sf
    err2_SF = (out[out['model_name']=='ML_SF']['y_true'].iloc[0].to_numpy() -
               out[out['model_name']=='ML_SF']['y_pred'].iloc[0].to_numpy())**2
    stat, p = shapiro(err2_noSF - err2_SF)
    if p > alphaW:
        print('Nat D is not normal')
    test = ttest_rel(err2_noSF, err2_SF, alternative="greater")
    # # do wilcoxon 2 (noSF vs SF)
    # test = d140_modelStats.paired_wilcoxon(err2_noSF, err2_SF, alternative="greater", alpha=alphaW, verbose=False)
    out.loc[(out['model_name'] == 'ML_SF'), 'wilcoxon'] = test[1]
    return out
def nat_level_comparison(config, model_mRes_dfs, model_names, crop, month_inSeas, hue_order, colors, dir_out):
    dirLTstats = os.path.join(config.data_dir, 'Label_analysis' + str(config.prct2retain))
    LTstats_fn = os.path.join(dirLTstats, config.AOI + '_' + 'Last5yrs' + 'Stats_retainPRCT' + str(config.prct2retain) + '.csv')
    df_stats = pd.read_csv(LTstats_fn)
    # get national level estimates by model and plot it
    plt.figure(figsize=(10, 6))
    firstLoop = True
    results = []
    for id, model in enumerate(model_names):
        mres = model_mRes_dfs[id]
        # merge with stats
        df_stats_crop = df_stats[df_stats['Crop_name|first'] == crop][['adm_id|', 'Area|mean']]
        mres = pd.merge(mres, df_stats_crop, left_on="adm_id", right_on="adm_id|", how="left")
        mres = mres.drop(columns=["adm_id|"])
        # mask both for a fair comparison
        mask = mres["yLoo_pred"].isna() | mres["yLoo_true"].isna()
        mres.loc[mask, ["yLoo_pred", "yLoo_true"]] = float("nan")
        yLoo_pred_nat = (mres.loc[mres["yLoo_pred"].notna() & mres["Area|mean"].notna() & (mres["Area|mean"] > 0)]
            .groupby("Year").apply(lambda g: np.average(g["yLoo_pred"], weights=g["Area|mean"]))
            .reset_index(name="yLoo_pred_weighted"))

        yLoo_true_nat = (mres.loc[mres["yLoo_true"].notna() & mres["Area|mean"].notna() & (mres["Area|mean"] > 0)]
                         .groupby("Year").apply(lambda g: np.average(g["yLoo_true"], weights=g["Area|mean"]))
                         .reset_index(name="yLoo_true_weighted"))
        # store error metric
        rmseNat = d140_modelStats.rmse_nan(yLoo_pred_nat['yLoo_pred_weighted'], yLoo_true_nat['yLoo_true_weighted'])
        rrmseNatprct = rmseNat / yLoo_true_nat['yLoo_true_weighted'].mean() * 100
        # first is true, second is estimated
        r2Nat = d140_modelStats.r2_nan(yLoo_true_nat['yLoo_true_weighted'], yLoo_pred_nat['yLoo_pred_weighted'])
        text = (f', $\mathrm{{RMSE}}_\mathrm{{p}}$={rmseNat:.2f}'
                f', $\mathrm{{rRMSE}}_\mathrm{{p}}$={rrmseNatprct:.2f}%'
                f', $\mathrm{{R}}^2_\mathrm{{p}}$={r2Nat:.2f}')
        # plot it
        if firstLoop:
            plt.plot(yLoo_true_nat['Year'], yLoo_true_nat['yLoo_true_weighted'], "--o", color='green', label="Observed")
            firstLoop = False
        ind = hue_order.index(model)
        plt.plot(yLoo_pred_nat['Year'], yLoo_pred_nat['yLoo_pred_weighted'], "-o", color=colors[ind],
                 label=model + text)
        #store also array of y pred and true for Wilcoxon test
        results.append({
            'model_name': model,
            'rrmseNatprct': rrmseNatprct,
            'r2Nat': r2Nat,
            'y_true': yLoo_true_nat['yLoo_true_weighted'],
            'y_pred': yLoo_pred_nat['yLoo_pred_weighted']
        })

    df_results = pd.DataFrame(results)
    plt.legend(prop={'size': 12}, loc='upper left', frameon=False)
    plt.xticks(np.arange(min(yLoo_true_nat['Year'] - 1), max(yLoo_true_nat['Year'] + 1), 1), rotation=40)
    units = pd.read_csv(os.path.join(config.data_dir, config.AOI + '_measurement_units.csv'))

    plt.ylabel("Yield [" + units['Yield'].iloc[0] + "]")
    title = 'Crop: ' + crop + ', Forecast time (month in season): ' +str(month_inSeas) + ' '

    plt.title(title)
    plt.tight_layout()
    fig_name = os.path.join(dir_out, 'National_' + crop + '_month_inSeas'  + str(month_inSeas) + '.png')
    plt.savefig(fig_name)
    plt.close()
    return df_results




if __name__ == '__main__':
    dir_out = os.path.join(baseDir, 'comp_' + rn1 + '_' + rn2 + '_' + rn3)
    os.makedirs(dir_out, exist_ok=True)
    mlsettings = a10_config.mlSettings(forecastingMonths=0)

    # get best model of each run

    configs = [a10_config.read(cf1, rn1), a10_config.read(cf2, rn2), a10_config.read(cf3, rn3)]

    # get calendar months
    calendarMonths = configs[0].forecastingCalendarMonths
    calendarMonths = ['01-' + calendar.month_abbr[(m % 12) + 1] for m in calendarMonths]

    # get start and end month sosMonth': 11, 'eosMonth': 5
    sosMonthCalendar = [x.sosMonth for x in configs]
    eosMonthCalendar = [x.eosMonth for x in configs]
    if (all(x == sosMonthCalendar[0] for x in sosMonthCalendar) == False) or \
    (all(x == eosMonthCalendar[0] for x in eosMonthCalendar) == False):
        print('Different dates in configs')
        sys.exit("Condition met, exiting program")
    else:
        sosMonthCalendar = sosMonthCalendar[0]
        eosMonthCalendar = eosMonthCalendar[0]
        eosMonthInSeason = configs[0].eosMonthInSeason
    runs = [rn1, rn2, rn3]
    short_names = [short_name1, short_name2, short_name3]
    df = pd.DataFrame()
    for run, config, short_name in zip(runs, configs, short_names):
        analysisOutputDir = os.path.join(config.models_out_dir, 'Analysis')
        b1 = pd.read_csv(analysisOutputDir + '/' + 'all_model_best1.csv')
        b1.insert(0, 'run_name', run)
        b1.insert(1, 'run_short_name', short_name)
        df = pd.concat([df, b1], ignore_index=True)
    list_time = df.forecast_time.unique()
    list_crop = df.Crop.unique()
    # replace name for plotting
    df["Estimator"] = df["Estimator"].replace("PeakNDVI", "PeakFPAR")

    # sanity check that  ["Null_model", "Trend", "PeakNDVI"] have same "rRMSE_p" at each 'forecast_time', no matter 'run_short_name'
    for crop in list_crop:
        for t in df.forecast_time.unique():
            for est in ["Null_model", "Trend", "PeakFPAR"]:
                tmp = df[(df['Crop'] == crop) & (df['forecast_time'] == t) & (df['Estimator'] == est)]
                if tmp["rRMSE_p"].nunique() != 1:
                    print(crop, t, est)
                    print("Not all rRMSE_p values are equal")

    ########################################################################################################
    # Bar of avg admin RMSE and R2 #########################################################################
    first_crop = True
    for crop in list_crop:
        print(crop)
        df2 = df.head(0) #make an empty df
        for t in df.forecast_time.unique():
            tmp = df[(df['Crop'] == crop) & (df['forecast_time'] == t)]
            # keep only one bennchmark
            for est in ["Null_model", "Trend", "PeakFPAR"]:
                tmp = pd.concat([
                    tmp[tmp['Estimator'] == est].drop_duplicates('Estimator', keep='first'),
                    tmp[tmp['Estimator'] != est]
                ]).sort_index().reset_index(drop=True)
            df2 = pd.concat([df2, tmp])
        df2['Estimator_plot'] = df2['Estimator'].map(lambda x: x if x in ['Null_model', 'PeakFPAR', 'Trend', 'Tab'] else 'ML')
        df2.loc[df2['Estimator_plot'] == 'ML', 'Estimator_plot'] = ('ML_' + df2.loc[df2['Estimator_plot'] == 'ML', 'run_short_name'])
        df2.loc[df2['Estimator_plot'] == 'Tab', 'Estimator_plot'] = ('Tab_' + df2.loc[df2['Estimator_plot'] == 'Tab', 'run_short_name'])

        hue_order = ["PeakFPAR", "ML_noSF", "Tab_noSF", "ML_ObsAsSF", "Tab_ObsAsSF", "ML_SF", "Tab_SF"]
        colors =    ["#8B2E2E",  "#1F3A5F", "#660066",  "#4A78A6",    "#660066",     "#A7C7E7", "#eb59eb"]

        if first_crop == True:
            df_all = df2
            first_crop = False
        else:
            df_all = pd.concat([df_all, df2], ignore_index=True)

        if plotTab != True:
            df2 = df2[df2['Estimator'] != 'Tab']
            indices = [i for i, s in enumerate(hue_order) if "Tab" not in s]
            hue_order = [hue_order[i] for i in indices]
            colors = [colors[i] for i in indices]
            suffix = '_no_Tab'
        else:
            suffix = ''
        if plotPeak != True:
            df2 = df2[df2['Estimator'] != 'PeakFPAR']
            indices = [i for i, s in enumerate(hue_order) if "PeakFPAR" not in s]
            hue_order = [hue_order[i] for i in indices]
            colors = [colors[i] for i in indices]
        # if crop == 'Soybeans':
        #     print()
        # remove null and trend from bar plot and plot them as horizontal lines (they do no change over time)
        nullRMSE = df2[df2['Estimator'] == 'Null_model']['rRMSE_p'].iloc[0]
        nullR2 = df2[df2['Estimator'] == 'Null_model']['avg_R2_p_temporal(alias R2_WITHINp)'].iloc[0]
        df2 = df2[df2['Estimator'] != 'Null_model']
        TrendRMSE = df2[df2['Estimator'] == 'Trend']['rRMSE_p'].iloc[0]
        TrendR2 = df2[df2['Estimator'] == 'Trend']['avg_R2_p_temporal(alias R2_WITHINp)'].iloc[0]
        df2 = df2[df2['Estimator'] != 'Trend']
        # Paired test
        df2 = (df2.groupby("forecast_time", group_keys=False).apply(compute_paired_test, runs=runs, configs=configs, short_names=short_names))
        # Figure
        x_order = df2.forecast_time.unique().tolist()
        palette = dict(zip(hue_order, colors))
        # Create vertically stacked subplots
        fig, (ax_top, ax_bottom) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True)
        fig.suptitle(crop, fontsize=16, fontweight="bold")
        # --- Top subplot: rRMSE_p ---
        sns.barplot(data=df2, x="forecast_time", y="rRMSE_p", hue="Estimator_plot", order=x_order, hue_order=hue_order, palette=palette, ax=ax_top)
        # Loop over the bars to add * for wiclcoxon
        for _, row in df2.iterrows():
            if not np.isnan(row['wilcoxon']) and row['wilcoxon'] < 0.05:
                x_val = row['forecast_time']
                hue_val = row['Estimator_plot']
                y_val = row['rRMSE_p']
                # Find the bar center, # In seaborn, bars are offset per hue, width=0.8/len(hue_order)
                n_hues = len(hue_order)
                bar_width = 0.8 / n_hues
                x_index = x_order.index(x_val)
                hue_index = hue_order.index(hue_val)
                x_coord = x_index - 0.4 + bar_width / 2 + hue_index * bar_width
                # Add the star slightly above the bar
                ax_top.text(x_coord, y_val + 0.01 * y_val, '*', ha='center', va='bottom', fontsize=12)

        ax_top.axhline(y=nullRMSE, color="grey", linestyle="--", linewidth=2, label="Null_model")
        ax_top.axhline(y=TrendRMSE, color="green", linestyle="--", linewidth=2, label="Trend")
        ax_top.set_title(r"Average rRMSE$_{\mathrm{p}}$ (%)")
        ax_top.set_xlabel("")  # remove x-label from top plot
        ax_top.legend(title="Estimator", loc="best")
        ax_top.set_ylabel(r"rRMSE$_{\mathrm{p}}$ (%)", fontsize=12)
        ax_top.set_xticklabels(calendarMonths)
        #ax_top.set_xlabel("Forecast time (month in season)", fontsize=12)
        ax_top.set_xlabel("Forecast time ", fontsize=12)

        # --- Bottom subplot: avg_R2_p_temporal (R2_WITHINp) ---
        sns.barplot(data=df2, x="forecast_time", y="avg_R2_p_temporal(alias R2_WITHINp)", hue="Estimator_plot", order=x_order, hue_order=hue_order, palette=palette, ax=ax_bottom)
        ax_bottom.axhline(0, color="0.3", linewidth=1)
        ax_bottom.set_ylim(min(ax_bottom.get_ylim()[0], 0), max(ax_bottom.get_ylim()[1], 0))
        ymin, ymax = ax_bottom.get_ylim()
        ax_bottom.axhline(y=nullR2, color="grey", linestyle="--", linewidth=2, label="Null_model")
        ax_bottom.axhline(y=TrendR2, color="green", linestyle="--", linewidth=2, label="Trend")
        new_ymin = min(ymin, TrendR2, nullR2)-0.05
        new_ymax = max(ymax, TrendR2, nullR2)
        ax_bottom.set_ylim(new_ymin, new_ymax)
        ax_bottom.set_title(r"Average $\mathrm{R}^2_{\mathrm{p}}$")
        ax_bottom.set_xticklabels(calendarMonths)
        ax_bottom.set_xlabel("Forecast time ", fontsize=12)
        ax_bottom.set_ylabel(r"$\mathrm{R}^2_{\mathrm{p}}$", fontsize=12)
        # --- Bar legend handles ---
        bar_handles = [Patch(facecolor=palette[est], edgecolor="black", label=est) for est in hue_order]
        # --- Line legend handles ---
        line_handles = [Line2D([0], [0], color="grey", linestyle="--", linewidth=2, label="Null_model"),
                        Line2D([0], [0], color="green", linestyle="--", linewidth=2, label="Trend")]
        # Combine both
        legend_handles = bar_handles + line_handles
        ax_top.legend(handles=legend_handles, title="Estimator", loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
        # Remove duplicate legend (keep only top one)
        ax_bottom.legend_.remove()
        fig_name = os.path.join(dir_out, crop + '_one_graph' + suffix + '.png')
        plt.tight_layout()
        plt.savefig(fig_name)
        plt.close()

    fn_name = os.path.join(dir_out, 'best_model_outputs.csv')
    df_all.to_csv(fn_name, index=False)
    print('Bar graph avg admin finished')
    # Bar of avg admin RMSE and R2 #########################################################################
    ########################################################################################################

    ########################################################################################################
    # Hindcasting pdf ######################################################################################
    # Now hindcasting by crop and admin (and Taylor, optiona;)
    # drop Tab
    df_all = df_all[df_all['Estimator'] != 'Tab']
    months_inSeas = [1, 2, 3, 4, 5]
    # get reg short_names
    df_regNames = pd.read_csv(os.path.join(configs[0].data_dir, config.AOI + '_REGION_id.csv'))
    allNatErrorStats = []
    for crop in list_crop:
        # loop on forecasting times
        print(crop)
        for month_inSeas in months_inSeas:
            df = df_all[(df_all['Crop'] == crop) & (df_all['forecast_time'] == month_inSeas)]
            # here I have six runs: nul, peak, trend, ML, Ml_ObsAsSF, ML_SF
            # model_names = df['Estimator_plot'].unique()
            model_names = ['ML_noSF', 'ML_ObsAsSF', 'ML_SF']
            # restrict analysis to ML
            model_mRes_dfs = []
            # get model outputs and ref
            for model in model_names:
                runID = df[df['Estimator_plot']==model]['runID'].values[0]
                run_name = df[df['Estimator_plot']==model]['run_name'].values[0]
                pos = runs.index(run_name)
                config = configs[pos]
                dir_of_out = config.models_out_dir
                rundID6digit = 'ID_' + str(runID).zfill(6)
                pattern = os.path.join(dir_of_out, f"{rundID6digit}*_mres.csv")
                file = glob.glob(pattern)
                if len(file) > 1:
                    print('problem: more than one mRes with given ID')
                mRes = pd.read_csv(file[0])
                mRes = mRes.merge(df_regNames[['adm_id', 'adm_name']], how='left', left_on='adm_id', right_on='adm_id')
                model_mRes_dfs.append(mRes)
            # now I have model names in list model_names and their mRes in model_mRes_dfs
            # get ref data
            ref = model_mRes_dfs[0]["yLoo_true"]
            # make sure that mRes files are aligned (same yLoo_true in the same year/adm_id order)
            all_identical = all(ref.equals(df["yLoo_true"]) for df in model_mRes_dfs[1:])
            if all_identical != True:
                print('Problem with yLoo_true')

            def nan_resistant_corr(g):
                mask = np.isfinite(g["yLoo_true"]) & np.isfinite(g["yLoo_pred"])
                if mask.sum() < 2:
                    return np.nan
                y_true = g.loc[mask, "yLoo_true"].to_numpy()
                y_pred = g.loc[mask, "yLoo_pred"].to_numpy()
                ccoef = np.corrcoef(y_pred, y_true)
                return ccoef[0]#[0, 1]

            if makeTaylor == True:
                ref_all = np.array(model_mRes_dfs[0]['yLoo_true'].values)
                mask_ref = np.isfinite(ref_all)
                # Calculate statistics for Taylor diagram
                models_all = np.array([i['yLoo_pred'].values for i in model_mRes_dfs])
                taylor_stats1 = sm.taylor_statistics(np.array(model_mRes_dfs[0]['yLoo_pred'].values)[mask_ref], ref_all[mask_ref])
                taylor_stats2 = sm.taylor_statistics(np.array(model_mRes_dfs[1]['yLoo_pred'].values)[mask_ref], ref_all[mask_ref])
                taylor_stats3 = sm.taylor_statistics(np.array(model_mRes_dfs[2]['yLoo_pred'].values)[mask_ref], ref_all[mask_ref])
                 # Store statistics in arrays
                sdev = np.array([taylor_stats1['sdev'][0], taylor_stats1['sdev'][1], taylor_stats2['sdev'][1], taylor_stats3['sdev'][1]])
                crmsd = np.array([taylor_stats1['crmsd'][0], taylor_stats1['crmsd'][1], taylor_stats2['crmsd'][1], taylor_stats3['crmsd'][1]])
                ccoef = np.array([taylor_stats1['ccoef'][0], taylor_stats1['ccoef'][1], taylor_stats2['ccoef'][1], taylor_stats3['ccoef'][1]])
                label = ['Obs'] + model_names
                intervalsCOR = np.concatenate((np.arange(0, 1.0, 0.2),  [0.9, 0.95, 0.99, 1]))
                intervalRMS = np.round(np.arange(0, 1.2, 0.2), 2)
                intervalSTD = np.arange(0, 2, 0.25)
                intervalRMS = np.arange(0, 26, 5)
                sm.taylor_diagram(sdev, crmsd, ccoef, styleOBS='-', colOBS='r', markerobs='o', titleOBS='Observation', labelrmspos='inside', markerLabel=label,  markerLegend='on')
                fn = os.path.join(dir_out, 'Taylor_' + crop + '_month_inSeas_' + str(month_inSeas) + '.png')
                plt.savefig(fn)
                plt.close()

            dfNatStats = nat_level_comparison(config, model_mRes_dfs, model_names, crop, month_inSeas, hue_order, colors, dir_out)
            dfNatStats['crop'] = crop
            dfNatStats['month_inSeas'] = month_inSeas
            allNatErrorStats.append(dfNatStats)
            # layout 1 loop over admin id (now I have model names in list model_names and their mRes in model_mRes_dfs)
            adm_ids = model_mRes_dfs[0]['adm_id'].unique()
            fig, axs = plt.subplots(max([len(adm_ids), 2]), 1, figsize=(10, 2.5 * len(adm_ids)))  # the max here is used because there might be just one admin, and subplot does not return axes
            axs = axs.flatten()
            axs_counter = 0

            # pdfs, I want to see the used forecast used (from forecast month to eos - 1 month)
            startCalendarMonth = sosMonthCalendar + month_inSeas # - 1
            if startCalendarMonth > 12:
                startCalendarMonth = startCalendarMonth - 12
            lastHorizon = eosMonthInSeason - month_inSeas - 1
            # print('Month in season: ' + str(month_inSeas)  + ', calendar month: ' + str(startCalendarMonth) + ', eosMonthCalendar: ' + str(eosMonthCalendar) + ', n months of seas:' + str(lastHorizon))
            df = S501_compare_season_obs_vs_SF_v2.get_seas_Obs_SF(cf2, cf3, startCalendarMonth, lastHorizon)
            # I have to assign a year (that of eos) to the extracted seas values
            if startCalendarMonth > eosMonthCalendar:
                df['Year'] = df['date'].dt.year + 1
            else:
                df['Year'] = df['date'].dt.year


            # layout 2 with correlation
            lw = 0.75
            n = len(adm_ids)
            fig = plt.figure(figsize=(10, 4 * n), constrained_layout=True)
            fig.subplots_adjust(right=0.6)
            gs = GridSpec(nrows=2 * n, ncols=2, height_ratios=[2, 2] * n, hspace=0.3, wspace=0.4)
            axes = []
            for i in range(n):
                # big plot (row 2*i, spans 2 columns)
                ax_big = fig.add_subplot(gs[2 * i, :])
                # two small plots (row 2*i + 1)
                ax_left = fig.add_subplot(gs[2 * i + 1, 0])
                ax_right = fig.add_subplot(gs[2 * i + 1, 1])
                ax_left.set_box_aspect(1)
                ax_right.set_box_aspect(1)
                axes.append((ax_big, ax_left, ax_right))

            # for i, (ax_big, ax_l, ax_r) in enumerate(axes):
            for i, (adm_id, (ax_big, ax_left, ax_right)) in enumerate(zip(adm_ids, axes)):
                # ax_big.plot([1, 2, 3], [i, i + 1, i + 2])
                # ax_big.set_title(f'Big plot {i + 1}')
                # plot ref
                ref_data = model_mRes_dfs[0][model_mRes_dfs[0]['adm_id'] == adm_id]
                merged_df = pd.merge(ref_data, df, on=['adm_id', 'Year'], how='left')
                ax_big.plot(ref_data['Year'], ref_data['yLoo_true'], "--o", color='green', label='Observed')
                ax_big2 = ax_big.twinx()
                prec = merged_df[merged_df['var'] == 'r']
                ax_big2.plot(prec['Year'], prec['obsMean_season'], "-", color='black', label='Prec_obsMean_season',
                         linewidth=lw)
                ax_big2.plot(prec['Year'], prec['sfMean_season'], "--", color='black', label='Prec_sfMean_season', linewidth=lw)
                ax_big2.set_ylabel('Prec season mean (mm/month')

                # Tertiary y-axis (offset)
                ax_big3 = ax_big.twinx()
                # divider = make_axes_locatable(ax_big)
                # ax_big3 = divider.append_axes("right", size="4%", pad=0.6)
                ax_big3.spines['right'].set_position(('outward', 30))  # shift right
                temp = merged_df[merged_df['var'] == 't']
                ax_big3.plot(temp['Year'], temp['obsMean_season'], "-", color='red', label='Temp_obsMean_season', linewidth=lw)
                ax_big3.plot(temp['Year'], temp['sfMean_season'], "--", color='red', label='Temp_sfMean_season', linewidth=lw)
                ax_big3.set_ylabel('Temp season mean (deg C)')
                # ax2.plot(merged_df['Year'], merged_df['yLoo_true'], "--o", color='green', label='Observed')
                # models
                for idx, model in enumerate(model_names):
                    model_data = model_mRes_dfs[idx][model_mRes_dfs[idx]['adm_id'] == adm_id]
                    if model == 'Trend':
                        color = 'green'
                    elif model == 'Null_model':
                        color = 'grey'
                    else:
                        ind = hue_order.index(model)
                        color = colors[ind]
                    r2 = d140_modelStats.r2_nan(model_data['yLoo_true'], model_data['yLoo_pred'])
                    ax_big.plot(model_data['Year'], model_data['yLoo_pred'], "-o", color=color,
                             label=model + ' R2p = ' + str(round(r2, 2)), linewidth=1.125, markersize=4.5)
                # ax1.legend(prop={'size': 6}, loc="upper left")
                ax_big.set_title(ref_data['adm_name'].iloc[0])
                ax_big.set_xlabel('Years')
                ax_big.set_ylabel('Yield [t/ha]')
                lines1, labels1 = ax_big.get_legend_handles_labels()
                lines2, labels2 = ax_big2.get_legend_handles_labels()
                lines3, labels3 = ax_big3.get_legend_handles_labels()
                ax_big.legend(
                    lines1 + lines2 + lines3,
                    labels1 + labels2 + labels3,
                    loc="upper left", prop={'size': 6}
                )
                # put yield in the foreground
                ax_big.set_zorder(3)
                ax_big2.set_zorder(2)
                ax_big3.set_zorder(1)
                # hide axis backgrounds
                ax_big.patch.set_visible(False)
                ax_big2.patch.set_visible(False)
                ax_big3.patch.set_visible(False)

                # raise ALL lines plotted on ax1
                for line in ax_big.lines:
                    line.set_zorder(10)

                # scatters
                x = prec['obsMean_season'].to_numpy()
                y = prec['yLoo_true'].to_numpy()
                ax_left.scatter(x, y, s=25,  alpha=0.8)
                # correlation coefficient (Pearson)
                mask = ~np.isnan(x) & ~np.isnan(y)
                r = np.corrcoef(x[mask], y[mask])[0, 1]
                # add text to plot
                ax_left.text(0.05, 0.95, f'r = {r:.2f}', transform=ax_left.transAxes, ha='left', va='top')
                # labels
                ax_left.set_xlabel('Prec obs mean (mm/month)')
                ax_left.set_ylabel('Yield (t/ha)')

                x = temp['obsMean_season'].to_numpy()
                y = prec['yLoo_true'].to_numpy()
                ax_right.scatter(x, y, s=25, alpha=0.8)
                # correlation coefficient (Pearson)
                mask = ~np.isnan(x) & ~np.isnan(y)
                r = np.corrcoef(x[mask], y[mask])[0, 1]
                # add text to plot
                ax_right.text(0.05, 0.95, f'r = {r:.2f}', transform=ax_right.transAxes, ha='left', va='top')
                # labels
                ax_right.set_xlabel('Temp obs mean (mm/month)')
                ax_right.set_ylabel('Yield (t/ha)')

            # fig.tight_layout()
            fn_name = os.path.join(dir_out, 'hindcasting_' + crop + '_month_inSeas_' + str(month_inSeas) + '_v2.pdf')
            plt.savefig(fn_name)
            plt.close()
    allNatErrorStats = pd.concat(allNatErrorStats, ignore_index=True)
    # Hindcasting pdf ######################################################################################
    ########################################################################################################

    ########################################################################################################
    # National level summary graph #########################################################################
    # National level summary graph
    for crop in list_crop:
        NatErrorStats = allNatErrorStats[allNatErrorStats['crop'] == crop]
        # Wilcoxon
        # if crop == 'Sunflower':
        #     print()
        NatErrorStats = (NatErrorStats.groupby("month_inSeas", group_keys=False).apply(compute_paired_test_nat))
        NatErrorStats = NatErrorStats.drop(columns=['y_true', 'y_pred'])
        # Figure
        x_order = NatErrorStats.month_inSeas.unique().tolist()
        if plotPeak != True:
            hue_order_nat = hue_order
            colors_nat = colors
        else:
            hue_order_nat = hue_order[1:]
            colors_nat = colors[1:]
        palette = dict(zip(hue_order_nat, colors_nat))
        # Create vertically stacked subplots
        fig, (ax_top, ax_bottom) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True)
        fig.suptitle(crop, fontsize=16, fontweight="bold")
        # --- Top subplot: rRMSE_p ---
        sns.barplot(data=NatErrorStats, x="month_inSeas", y="rrmseNatprct", hue="model_name", order=x_order, hue_order=hue_order_nat,
                    palette=palette, ax=ax_top)
        # Loop over the bars to add * for wiclcoxon
        for _, row in NatErrorStats.iterrows():
            if not np.isnan(row['wilcoxon']) and row['wilcoxon'] < 0.05:
                x_val = row['month_inSeas']
                hue_val = row['model_name']
                y_val = row['rrmseNatprct']

                # Find the bar center
                # In seaborn, bars are offset per hue, width=0.8/len(hue_order)
                n_hues = len(hue_order_nat)
                bar_width = 0.8 / n_hues
                x_index = x_order.index(x_val)
                hue_index = hue_order_nat.index(hue_val)
                x_coord = x_index - 0.4 + bar_width / 2 + hue_index * bar_width

                # Add the star slightly above the bar
                ax_top.text(x_coord, y_val + 0.01 * y_val, '*', ha='center', va='bottom', fontsize=12)

        # ax_top.axhline(y=nullRMSE, color="grey", linestyle="--", linewidth=2, label="Null_model")
        # ax_top.axhline(y=TrendRMSE, color="green", linestyle="--", linewidth=2, label="Trend")
        ax_top.set_title(r"rRMSE$_{\mathrm{p}}$ (%)")
        ax_top.set_xlabel("")  # remove x-label from top plot
        ax_top.legend(title="Estimator", loc="best")
        ax_top.set_ylabel(r"rRMSE$_{\mathrm{p}}$ (%)", fontsize=12)
        ax_top.set_xticklabels(calendarMonths)
        ax_top.set_xlabel("Forecast time ", fontsize=12)
        # ax_top.set_xlabel("Forecast time (month in season)", fontsize=12)

        # --- Bottom subplot: avg_R2_p_temporal (R2_WITHINp) ---
        sns.barplot(data=NatErrorStats, x="month_inSeas", y="r2Nat", hue="model_name",
                    order=x_order, hue_order=hue_order_nat, palette=palette, ax=ax_bottom)
        ax_bottom.axhline(0, color="0.3", linewidth=1)
        ax_bottom.set_ylim(min(ax_bottom.get_ylim()[0], 0), max(ax_bottom.get_ylim()[1], 0))
        ax_bottom.set_title(r"$\mathrm{R}^2_{\mathrm{p}}$")
        # ax_bottom.set_xlabel("Forecast time (month in season)")
        ax_bottom.set_xticklabels(calendarMonths)
        ax_bottom.set_xlabel("Forecast time ", fontsize=12)
        ax_bottom.set_ylabel(r"$\mathrm{R}^2_{\mathrm{p}}$", fontsize=12)

        # --- Bar legend handles ---
        bar_handles = [Patch(facecolor=palette[est], edgecolor="black", label=est) for est in hue_order_nat]


        # Combine both
        legend_handles = bar_handles #+ line_handles

        ax_top.legend(handles=legend_handles, title="Estimator", loc="upper left", bbox_to_anchor=(1.02, 1),
                      borderaxespad=0.0)

        # Remove duplicate legend (keep only top one)
        ax_bottom.legend_.remove()
        fig_name = os.path.join(dir_out, crop + '_one_graph_national' + suffix + '.png')
        plt.tight_layout()
        plt.savefig(fig_name)
        plt.close()

    # National level summary graph #########################################################################
    ########################################################################################################

    print('End of comparison')
