import pandas as pd
from A_config import a10_config
import os
import sys
import glob
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
from F_post_processsing import F100_analyze_hindcast_output
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

baseDir = r'V:\foodsec\Projects\SNYF\SIDvs\ZA\summer2025data\SF'
cf1 = os.path.join(baseDir, 'SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config1235.json')
rn1 = 'ZA_NoSF'
short_name1 = 'noSF'

cf2 = os.path.join(baseDir, 'SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config1235_ObsAsForecast.json')
rn2 = 'ZA_ObsAsSF'
short_name2 = 'ObsAsSF'

cf3 = os.path.join(baseDir, 'SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config1235_SfAsForecast.json')
rn3 = 'ZA_SfAsSF'
short_name3 = 'SF'

# plot Tab results or not
plotTab = False
# Plot Taylor
makeTaylor = False
########################################################




dir_out = os.path.join(baseDir, 'comp_' + rn1 + '_' + rn2 + '_' + rn3)
os.makedirs(dir_out, exist_ok=True)
mlsettings = a10_config.mlSettings(forecastingMonths=0)
# to gather: list of times, list of crops; a df with:
# run_name, time, crop, model (get all for now), avg rmse_perct, avg r2 by admin, same at national lvel

# get best model of each run
df = pd.DataFrame()
configs = [a10_config.read(cf1, rn1), a10_config.read(cf2, rn2), a10_config.read(cf3, rn3)]

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

first_crop = True
for crop in list_crop:
    df2 = df.head(0) #make an empty
    for t in df.forecast_time.unique():
        tmp = df[(df['Crop'] == crop) & (df['forecast_time'] == t)]
        print()
        # keep only one bennchmark
        for est in ["Null_model", "Trend", "PeakFPAR"]:
            tmp = pd.concat([
                tmp[tmp['Estimator'] == est].drop_duplicates('Estimator', keep='first'),
                tmp[tmp['Estimator'] != est]
            ]).sort_index().reset_index(drop=True)
        df2 = pd.concat([df2, tmp])
    df2['Estimator_plot'] = df2['Estimator'].map(lambda x: x if x in ['Null_model', 'PeakFPAR', 'Trend', 'Tab'] else 'ML')
    df2.loc[df2['Estimator_plot'] == 'ML', 'Estimator_plot'] = (
        'ML_' + df2.loc[df2['Estimator_plot'] == 'ML', 'run_short_name']
    )
    df2.loc[df2['Estimator_plot'] == 'Tab', 'Estimator_plot'] = (
            'Tab_' + df2.loc[df2['Estimator_plot'] == 'Tab', 'run_short_name']
    )

    hue_order = ["PeakFPAR", "ML_noSF", "Tab_noSF", "ML_ObsAsSF", "Tab_ObsAsSF", "ML_SF", "Tab_SF"]
    colors = [   "#8B2E2E", "#1F3A5F", "#660066",   "#4A78A6",    "#660066",   "#A7C7E7", "#eb59eb"]
    # colors = ["#FF0000", "#0000FF", "#660066", "#4da2e8", "#660066", "#42d1f5", "#eb59eb"]


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
    # remove null and trend from bar plot and plot them as horizontal lines (they do no change over time)
    nullRMSE = df2[df2['Estimator'] == 'Null_model']['rRMSE_p'].iloc[0]
    nullR2 = df2[df2['Estimator'] == 'Null_model']['avg_R2_p_temporal(alias R2_WITHINp)'].iloc[0]
    df2 = df2[df2['Estimator'] != 'Null_model']
    TrendRMSE = df2[df2['Estimator'] == 'Trend']['rRMSE_p'].iloc[0]
    TrendR2 = df2[df2['Estimator'] == 'Trend']['avg_R2_p_temporal(alias R2_WITHINp)'].iloc[0]
    df2 = df2[df2['Estimator'] != 'Trend']
    # Figure
    x_order = df2.forecast_time.unique().tolist()
    palette = dict(zip(hue_order, colors))
    # Create vertically stacked subplots
    fig, (ax_top, ax_bottom) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True)
    fig.suptitle(crop, fontsize=16, fontweight="bold")
    # --- Top subplot: rRMSE_p ---
    sns.barplot(data=df2, x="forecast_time", y="rRMSE_p", hue="Estimator_plot", order=x_order, hue_order=hue_order, palette=palette, ax=ax_top)
    ax_top.axhline(y=nullRMSE, color="grey", linestyle="--", linewidth=2, label="Null_model")
    ax_top.axhline(y=TrendRMSE, color="green", linestyle="--", linewidth=2, label="Trend")
    ax_top.set_title("rRMSEp")
    ax_top.set_xlabel("")  # remove x-label from top plot
    ax_top.legend(title="Estimator", loc="best")
    ax_top.set_ylabel("rRMSEp (%)", fontsize=12)
    ax_top.set_xlabel("Forecast time (month in season)", fontsize=12)

    # --- Bottom subplot: avg_R2_p_temporal (R2_WITHINp) ---
    sns.barplot(data=df2, x="forecast_time", y="avg_R2_p_temporal(alias R2_WITHINp)", hue="Estimator_plot", order=x_order, hue_order=hue_order, palette=palette, ax=ax_bottom)
    ax_bottom.axhline(0, color="0.3", linewidth=1)
    ax_bottom.set_ylim(min(ax_bottom.get_ylim()[0], 0), max(ax_bottom.get_ylim()[1], 0))
    ax_bottom.axhline(y=nullR2, color="grey", linestyle="--", linewidth=2, label="Null_model")
    ax_bottom.axhline(y=TrendR2, color="green", linestyle="--", linewidth=2, label="Trend")
    ax_bottom.set_title("Average R2p")
    ax_bottom.set_xlabel("Forecast time (month in season)")
    ax_bottom.set_ylabel("R2p", fontsize=12)

    # --- Bar legend handles ---
    bar_handles = [Patch(facecolor=palette[est], edgecolor="black", label=est) for est in hue_order]

    # --- Line legend handles ---
    line_handles = [
        Line2D([0], [0], color="grey", linestyle="--", linewidth=2, label="Null_model"),
        Line2D([0], [0], color="green", linestyle="--", linewidth=2, label="Trend")
    ]

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
print('First plotting finished')
# Now hindcasting by crop and admin (and Taylor?, corr is overall in laylor, unless I give the avg corr, but would be strange)
# drop Tab
df_all = df_all[df_all['Estimator'] != 'Tab']
months_inSeas = [1, 2, 3]
# get reg short_names
df_regNames = pd.read_csv(os.path.join(configs[0].data_dir, config.AOI + '_REGION_id.csv'))
# loop on crops
# list_crop = ['Maize_white']
# months_inSeas = [2]
for crop in list_crop:
    # loop on forecasting times
    for month_inSeas in months_inSeas:
        df = df_all[(df_all['Crop'] == crop) & (df_all['forecast_time'] == month_inSeas)]
        # here I have six runs: nul, peak, trend, ML, Ml_ObsAsSF, ML_SF
        model_names = df['Estimator_plot'].unique()
        model_names = ['ML_noSF', 'ML_ObsAsSF', 'ML_SF']
        # restrict analysis
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
        # Taylor diagram
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
            # replace corr (ccoef = np.corrcoef(p,r); ccoef = ccoef[0]) as avg at admin level
            # taylor_stats1['ccoef'] = (model_mRes_dfs[0].groupby("adm_id").apply(nan_resistant_corr)).mean()
            taylor_stats2 = sm.taylor_statistics(np.array(model_mRes_dfs[1]['yLoo_pred'].values)[mask_ref], ref_all[mask_ref])
            # taylor_stats2['ccoef'] = (model_mRes_dfs[1].groupby("adm_id").apply(nan_resistant_corr)).mean()
            taylor_stats3 = sm.taylor_statistics(np.array(model_mRes_dfs[2]['yLoo_pred'].values)[mask_ref], ref_all[mask_ref])
            # taylor_stats3['ccoef'] = (model_mRes_dfs[2].groupby("adm_id").apply(nan_resistant_corr)).mean()
            # Store statistics in arrays
            sdev = np.array([taylor_stats1['sdev'][0], taylor_stats1['sdev'][1], taylor_stats2['sdev'][1], taylor_stats3['sdev'][1]])
            crmsd = np.array([taylor_stats1['crmsd'][0], taylor_stats1['crmsd'][1], taylor_stats2['crmsd'][1], taylor_stats3['crmsd'][1]])
            ccoef = np.array([taylor_stats1['ccoef'][0], taylor_stats1['ccoef'][1], taylor_stats2['ccoef'][1], taylor_stats3['ccoef'][1]])
            # Normalize by reference STD
            # sdev = sdev / sdev[0]
            label = ['Obs'] + model_names
            intervalsCOR = np.concatenate((np.arange(0, 1.0, 0.2),
                                           [0.9, 0.95, 0.99, 1]))
            intervalRMS = np.round(np.arange(0, 1.2, 0.2), 2)
            intervalSTD = np.arange(0, 2, 0.25)
            intervalRMS = np.arange(0, 26, 5)
            sm.taylor_diagram(sdev, crmsd, ccoef, styleOBS='-',
                              colOBS='r', markerobs='o',
                              titleOBS='Observation', labelrmspos='inside',
                              markerLabel=label,  markerLegend='on')


            fn = os.path.join(dir_out, 'Taylor_' + crop + '_month_inSeas_' + str(month_inSeas) + '.png')
            plt.savefig(fn)
            plt.close()
            # Keep only where all have finite values
            # mask_ref = np.isfinite(ref_all)
            # mask_models = np.isfinite(models_all).all(axis=0)
            # mask = mask_ref & mask_models


            print()

        # layout 1 loop over admin id
        adm_ids = model_mRes_dfs[0]['adm_id'].unique()
        fig, axs = plt.subplots(max([len(adm_ids), 2]), 1, figsize=(10, 2.5 * len(adm_ids)))  # the max here is used because there might be just one admin, and subplot does not return axes
        axs = axs.flatten()
        axs_counter = 0

        # get pheno, cumulated and sanity check,
        # S501_compare_season_obs_vs_SF_v2.get_seas_Obs_SF(config_path_ObsAsSF, config_path_SF, startCalendarMonth, lastHorizon)
        # now I am in month_inSeas, sosMonthCalendar
        # I want to see the used  forecast used (from now to eos - 1 month)
        startCalendarMonth = sosMonthCalendar + month_inSeas - 1
        if startCalendarMonth > 12:
            startCalendarMonth = startCalendarMonth -12
        lastHorizon = eosMonthInSeason - month_inSeas - 1
        print('Month in season: ' + str(month_inSeas)  + ', calendar month: ' + str(startCalendarMonth) + ', eosMonthCalendar: ' + str(eosMonthCalendar) + ', n months of seas:' + str(lastHorizon))
        df = S501_compare_season_obs_vs_SF_v2.get_seas_Obs_SF(cf2, cf3, startCalendarMonth, lastHorizon)
        # I have to assign a year (that of eos) to the extracted seas values
        if startCalendarMonth > eosMonthCalendar:
            df['Year'] = df['date'].dt.year + 1
        else:
            df['Year'] = df['date'].dt.year
        # for adm_id in adm_ids:
        #     lw = 0.75
        #     ax1  = axs[axs_counter]
        #     # plot ref
        #     ref_data = model_mRes_dfs[0][model_mRes_dfs[0]['adm_id']==adm_id]
        #     merged_df = pd.merge(ref_data, df, on=['adm_id', 'Year'], how='left')
        #     ax1.plot(ref_data['Year'], ref_data['yLoo_true'], "--o", color='green', label='Observed')
        #     ax2 = ax1.twinx()
        #     prec = merged_df[merged_df['var'] == 'r']
        #
        #     ax2.plot(prec['Year'], prec['obsMean_season'], "-", color='black', label='Prec_obsMean_season', linewidth=lw)
        #     ax2.plot(prec['Year'], prec['sfMean_season'], "--", color='black', label='Prec_sfMean_season', linewidth=lw)
        #     ax2.set_ylabel('Prec season mean (mm/month')
        #
        #     # Tertiary y-axis (offset)
        #     ax3 = ax1.twinx()
        #     ax3.spines['right'].set_position(('outward', 60))  # shift right
        #     temp = merged_df[merged_df['var'] == 't']
        #     ax3.plot(temp['Year'], temp['obsMean_season'], "-", color='red', label='Temp_obsMean_season', linewidth=lw)
        #     ax3.plot(temp['Year'], temp['sfMean_season'], "--", color='red', label='Temp_sfMean_season', linewidth=lw)
        #     ax3.set_ylabel('Temp season mean (deg C)')
        #     # ax2.plot(merged_df['Year'], merged_df['yLoo_true'], "--o", color='green', label='Observed')
        #     # models
        #     for idx, model in enumerate(model_names):
        #         model_data = model_mRes_dfs[idx][model_mRes_dfs[idx]['adm_id']==adm_id]
        #         if model == 'Trend':
        #             color = 'green'
        #         elif model == 'Null_model':
        #             color = 'grey'
        #         else:
        #             ind = hue_order.index(model)
        #             color = colors[ind]
        #         r2 = d140_modelStats.r2_nan(model_data['yLoo_true'] ,model_data['yLoo_pred'])
        #         ax1.plot(model_data['Year'], model_data['yLoo_pred'], "-o", color=color, label=model + ' R2p = ' + str(round(r2, 2)), linewidth=1.125, markersize=4.5)
        #     # ax1.legend(prop={'size': 6}, loc="upper left")
        #     ax1.set_title(ref_data['adm_name'].iloc[0])
        #     ax1.set_xlabel('Years')
        #     ax1.set_ylabel('Yield [t/ha]')
        #     axs_counter = axs_counter + 1
        #     lines1, labels1 = ax1.get_legend_handles_labels()
        #     lines2, labels2 = ax2.get_legend_handles_labels()
        #     lines3, labels3 = ax3.get_legend_handles_labels()
        #     ax1.legend(
        #         lines1 + lines2 + lines3,
        #         labels1 + labels2 + labels3,
        #         loc="upper left", prop={'size': 6}
        #     )
        #     # put yield in the foreground
        #     ax1.set_zorder(3)
        #     ax2.set_zorder(2)
        #     ax3.set_zorder(1)
        #     # hide axis backgrounds
        #     ax1.patch.set_visible(False)
        #     ax2.patch.set_visible(False)
        #     ax3.patch.set_visible(False)
        #
        #     # raise ALL lines plotted on ax1
        #     for line in ax1.lines:
        #         line.set_zorder(10)
        # fig.tight_layout()
        # fn_name = os.path.join(dir_out, 'hindcasting_' + crop + '_month_inSeas_' + str(month_inSeas) + '.pdf')
        # plt.savefig(fn_name)
        # plt.close()

        # layout 2 with correlation
        lw = 0.75
        n = len(adm_ids)
        fig = plt.figure(figsize=(10, 4 * n), constrained_layout=True)
        fig.subplots_adjust(right=0.6)
        # fig.subplots_adjust(
        #     top=0.95,
        #     bottom=0.05
        # )
        # gs = GridSpec(
        #     nrows=2 * n,
        #     ncols=2,
        #     height_ratios=[2, 1] * n,  # big plot taller
        #     hspace=0.4,
        #     wspace=0.3
        # )
        gs = GridSpec(
            nrows=2 * n,
            ncols=2,
            height_ratios=[2, 2] * n,  # ⬅ second row same height as column width
            hspace=0.3,
            wspace=0.4 #0.2
        )
        # gs = GridSpec(
        #     nrows=3 * n,
        #     ncols=2,
        #     height_ratios=[2, 2, 0.3] * n,  # big, square, spacer
        #     hspace=0.0,
        #     wspace=0.25
        # )
        axes = []
        for i in range(n):
            # big plot (row 2*i, spans 2 columns)
            ax_big = fig.add_subplot(gs[2 * i, :])
            # two small plots (row 2*i + 1)
            ax_left = fig.add_subplot(gs[2 * i + 1, 0])
            ax_right = fig.add_subplot(gs[2 * i + 1, 1])
            ax_left.set_box_aspect(1)
            ax_right.set_box_aspect(1)
            # row = 3 * i
            #
            # ax_big = fig.add_subplot(gs[row, :])
            #
            # ax_left = fig.add_subplot(gs[row + 1, 0])
            # ax_right = fig.add_subplot(gs[row + 1, 1])
            #
            # # square bottom plots
            # ax_left.set_box_aspect(1)
            # ax_right.set_box_aspect(1)
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

