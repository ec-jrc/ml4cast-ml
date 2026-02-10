import pandas as pd
from A_config import a10_config
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
import glasbey
import math

"""
Compare seasonal cumulation of obs and SF 
"""
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', None)

# configuration files and run names
# TN
# baseDir = r'V:\foodsec\Projects\SNYF\SIDv\TN\SF'
# cf1 = os.path.join(baseDir, 'ObsAsSF\TNMultiple_WC-Tunisia-ASAP_config_ObsAsForecast.json')
# cf2 = os.path.join(baseDir, 'SF\TNMultiple_WC-Tunisia-ASAP_config_SfAsForecast.json')
# # seasonal cumulation goes from .. to .. (in calendar month)
# startMonth = 12 #10

# # remember that forecast in month 10 with horizon 1 is that of 11,
# # so in month 10 up to 5 I have to take horizon 1 (11) to 7 (5)
# lastHorizon = 1 #7
# ZA
baseDir = r'V:\foodsec\Projects\SNYF\SIDvs\ZA\summer2025data'
cf1 = os.path.join(baseDir, 'SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config1235_ObsAsForecast.json')
cf2 = os.path.join(baseDir, 'SF_test_ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config1235_SfAsForecast.json')
# seasonal cumulation goes from .. to .. (in calendar month)
startMonth = 3 #10

# remember that forecast in month 10 with horizon 1 is that of 11,
# so in month 10 up to 5 I have to take horizon 1 (11) to 7 (5)
lastHorizon = 2 #7

prefix = 'M' + str(startMonth) + "_Horizon" + str(lastHorizon)
########################################################

def plot_by_adm_and_time(df, fn):
    """
    Plot time series of sfMean_season and obsMean_season by adm_id.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
        ['adm_id', 'date', 'sf', 'obs']
    """

    # Ensure date is datetime
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    fig, ax = plt.subplots(figsize=(10, 6))
    # Create a color cycle explicitly
    adm_ids = sorted(df['adm_id'].unique())
    colors = plt.cm.tab10(range(len(adm_ids)))
    color_map = dict(zip(adm_ids, colors))
    for adm_id, g in df.groupby('adm_id'):
        g = g.sort_values('date')
        c = color_map[adm_id]
        # sfMean_season: solid line
        ax.plot(
            g['date'],
            g['sf'],
            color=c,
            linestyle='-',
            label=f'{adm_id} sfMean'
        )

        # obsMean_season: dashed line
        ax.plot(
            g['date'],
            g['obs'],
            color=c,
            linestyle='--',
            label=f'{adm_id} obsMean'
        )

    ax.set_xlabel('Date')
    ax.set_ylabel('Seasonal Mean')
    ax.set_title('sfMean_season vs obsMean_season by adm_id')

    # Avoid duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(),
              dict(zip(labels, handles)).keys(),
              loc='best')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches='tight')

def plot_by_adm_and_time_subplots(df, fn):
    """
    Plot sfMean_season and obsMean_season time series,
    one subplot per adm_id.
    """

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    adm_ids = sorted(df['adm_id'].unique())
    n = len(adm_ids)

    # Layout: up to 2 columns, adjust rows automatically
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12, 4 * nrows),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    for ax, adm_id in zip(axes, adm_ids):
        g = df[df['adm_id'] == adm_id].sort_values('date')

        # Single color per subplot
        color = 'tab:blue'

        ax.plot(
            g['date'],
            g['sf'],
            color=color,
            linestyle='-',
            label='sfMean_season'
        )

        ax.plot(
            g['date'],
            g['obs'],
            color=color,
            linestyle='--',
            label='obsMean_season'
        )

        ax.set_title(f'adm_id: {adm_id}')
        ax.grid(True, alpha=0.3)

    # Remove unused subplots if any
    for ax in axes[len(adm_ids):]:
        ax.axis('off')

    axes[0].legend(loc='best')

    fig.supxlabel('Date')
    fig.supylabel('Seasonal Mean')
    fig.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches='tight')

def plot(df, fn, option='', colorby='year'):
    df["obs"] = pd.to_numeric(df["obs"], errors="coerce")
    df["sf"] = pd.to_numeric(df["sf"], errors="coerce")
    if colorby=='year':
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["color"] = df["date"].dt.year
        color_title = 'Year'
    elif colorby=='adm_id':
        df["color"] = df["adm_id"]
        color_title = 'adm_id'

    color_sorted = sorted(df["color"].unique())
    if option == 'z':
        df['obs'] = (df['obs'] - df['obs'].mean()) / df['obs'].std()
        df['sf'] = (df['sf'] - df['sf'].mean()) / df['sf'].std()

    # Drop rows where either value is missing
    df_clean = df.dropna(subset=["obs", "sf"])
    r, p_val = pearsonr(df_clean["obs"], df_clean["sf"])
    # Format the text that will appear on the plot
    corr_text = f"$r$ = {r:.3f}\n$p$ = {p_val:.3g}"
    plt.figure(figsize=(8, 6))
    # cmap = plt.get_cmap("tab20")  # first 20
    # colors1 = [cmap(i) for i in range(20)]
    # cmap2 = plt.get_cmap("tab10")  # add 10 more distinct colors
    # colors2 = [cmap2(i) for i in range(10)]
    # # Combine and trim to 25
    # palette_25 = colors1 + colors2
    # palette_25 = palette_25[:25]
    palette_25 = glasbey.create_palette(palette_size=len(color_sorted))
    sns.scatterplot(
        data=df_clean,
        x="obs",
        y="sf",
        hue="color",
        hue_order=color_sorted,  # <-- ensures ALL years appear
        palette=palette_25, #"viridis",
        s=60,
        edgecolor="k",
        linewidth=0.5,
        alpha=0.8,
        legend="full"
    )
    plt.text(
        0.02, 0.98,  # x‑, y‑position in axes fraction coordinates
        corr_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3", alpha=0.8)
    )
    overall_min = min(df_clean["obs"].min(), df_clean["sf"].min())
    overall_max = max(df_clean["obs"].max(), df_clean["sf"].max())
    margin = (overall_max - overall_min) * 0.02
    xlim = (overall_min - margin, overall_max + margin)
    # 1:1 line (y = x)
    plt.plot(xlim, xlim, color="red", lw=2, ls="--", label="1:1 line")
    # Apply the same limits to both axes
    plt.xlim(xlim)
    plt.ylim(xlim)
    # Keep the aspect ratio truly square (data‑units are equal)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Scatter plot of obs vs. sf with Pearson correlation")
    plt.xlabel("obs")
    plt.ylabel("sf")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title=color_title, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches='tight')


def get_sea_Obs_SF(congig_path_ObsAsSF, congig_path_SF, startMonth, lastHorizon):
    config_ObsAsSF = a10_config.read(congig_path_ObsAsSF, "")
    config_SF = a10_config.read(congig_path_SF, "")
    dfo = pd.read_csv(os.path.join(config_ObsAsSF.data_dir, config_ObsAsSF.afi + '.csv'))
    dfo = dfo.rename(columns={'mean': 'obsMean'})
    dfs = pd.read_csv(os.path.join(config_SF.data_dir, config_SF.afi + '.csv'))
    dfs = dfs.rename(columns={'mean': 'sfMean'})
    # merge versions
    df = pd.merge(dfs, dfo, on=['adm_id', 'variable_name', 'date'])
    # keep only forecast
    df = df[df["variable_name"].str.contains("SF", na=False, case=False)]
    df[['SF', 'var', 'horizon']] = df['variable_name'].str.split('_', n=2, expand=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    mask = df["date"].dt.month == startMonth  # Boolean Series
    df_month = df.loc[mask].copy()
    df_month["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
    df_month = df_month.reset_index(drop=True)
    mask = df_month["horizon"].between(1, lastHorizon, inclusive="both")  # pandas ≥1.3 uses inclusive="both"
    df_filt = df_month.loc[mask].copy()
    # df_filt.to_csv(os.path.join(dir_out, prefix + 'test_merge.csv'), index=False)
    agg = (
        df_filt
        .groupby(["adm_id", "date", "var"], as_index=False)  # keep keys as columns
        .agg(sfMean_season=("sfMean", "mean"), obsMean_season=("obsMean", "mean"),
             )  # rename the aggregated column to something clear
    )
    return agg

agg = get_sea_Obs_SF(cf1, cf2, startMonth, lastHorizon)
dir_out = os.path.join(baseDir, 'comp_seasonal_obs_SF')
os.makedirs(dir_out, exist_ok=True)

# config1 = a10_config.read(cf1, "")
# config2 = a10_config.read(cf2, "")
#
# dir_out = os.path.join(baseDir, 'comp_seasonal_obs_SF')
# os.makedirs(dir_out, exist_ok=True)
# mlsettings = a10_config.mlSettings(forecastingMonths=0)
#
# dfo = pd.read_csv(os.path.join(config1.data_dir, config1.afi + '.csv'))
# dfo = dfo.rename(columns={'mean': 'obsMean'})
#
# dfs = pd.read_csv(os.path.join(config2.data_dir, config2.afi + '.csv'))
# dfs = dfs.rename(columns={'mean': 'sfMean'})
#
# # merge version
# df = pd.merge(dfs, dfo, on=['adm_id', 'variable_name', 'date'])
# # df = df.drop('x', axis=1)
# # keep only forecast
# df = df[df["variable_name"].str.contains("SF", na=False, case=False)]
# df[['SF', 'var', 'horizon']] = df['variable_name'].str.split('_', n=2, expand=True)
# # df.to_csv(os.path.join(dir_out, 'test_merge.csv'), index=False)
# df["date"] = pd.to_datetime(df["date"], errors="coerce")
# mask = df["date"].dt.month == startMonth      # Boolean Series
# df_month = df.loc[mask].copy()
# df_month["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
# df_month = df_month.reset_index(drop=True)
# mask = df_month["horizon"].between(1, lastHorizon, inclusive="both")   # pandas ≥1.3 uses inclusive="both"
# df_filt = df_month.loc[mask].copy()
# df_filt.to_csv(os.path.join(dir_out, prefix + 'test_merge.csv'), index=False)
# agg = (
#     df_filt
#     .groupby(["adm_id", "date", "var"], as_index=False)   # keep keys as columns
#     .agg(sfMean_season=("sfMean", "mean"), obsMean_season=("obsMean", "mean"),
#     )   # rename the aggregated column to something clear
# )
agg.to_csv(os.path.join(dir_out, prefix + 'test_agg_merge.csv'), index=False)
agg = agg.rename(columns={"obsMean_season": "obs", "sfMean_season": "sf"})

#Plotting
# plot(agg[agg["var"] == 'r'][["obs", "sf", "date"]], os.path.join(dir_out, prefix + '_rain_corr_merge.png'), option='')
# plot(agg[agg["var"] == 't'][["obs", "sf", "date"]], os.path.join(dir_out, prefix + '_temp_corr_merge.png'), option='')
# # now z
# plot(agg[agg["var"] == 'r'][["obs", "sf", "date"]], os.path.join(dir_out, prefix + '_z_rain_corr_merge_year.png'), option='z', colorby='year')
# plot(agg[agg["var"] == 't'][["obs", "sf", "date"]], os.path.join(dir_out, prefix + '_z_temp_corr_merge_year.png'), option='z', colorby='year')

plot(agg[agg["var"] == 'r'][["obs", "sf", "adm_id"]], os.path.join(dir_out, prefix + '_z_rain_corr_merge_adm_id.png'), option='z', colorby='adm_id')
plot(agg[agg["var"] == 't'][["obs", "sf", "adm_id"]], os.path.join(dir_out, prefix + '_z_temp_corr_merge_adm_id.png'), option='z', colorby='adm_id')
plot_by_adm_and_time(agg[agg["var"] == 'r'], os.path.join(dir_out, prefix + '_rain_time_series.png'))
plot_by_adm_and_time_subplots(agg[agg["var"] == 'r'], os.path.join(dir_out, prefix + '_rain_time_series_subplots.png'))


