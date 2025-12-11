import pandas as pd
from A_config import a10_config
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
"""
Compare seasonal cumulation of obs and SF 
"""

# configuration files and run names
baseDir = r'V:\foodsec\Projects\SNYF\SIDv\TN\SF'
cf1 = os.path.join(baseDir, 'ObsAsSF\TNMultiple_WC-Tunisia-ASAP_config_ObsAsForecast.json')
cf2 = os.path.join(baseDir, 'SF\TNMultiple_WC-Tunisia-ASAP_config_SfAsForecast.json')
# seasonal cumulation goes from .. to .. (in calendar month)
startMonth = 10
endMonth = 5
# remember that forecast in month 10 with horizon 1 is that of 11,
# so in month 10 up to 5 I have to take horizon 1 (11) to 7 (5)
lastHorizon = 7
########################################################


config1 = a10_config.read(cf1, "")
config2 = a10_config.read(cf2, "")

dir_out = os.path.join(baseDir, 'comp_seasonal_obs_SF')
os.makedirs(dir_out, exist_ok=True)
mlsettings = a10_config.mlSettings(forecastingMonths=0)

dfo = pd.read_csv(os.path.join(config1.data_dir, config1.afi + '.csv'))
dfo["type"] = "obs"
dfs = pd.read_csv(os.path.join(config2.data_dir, config2.afi + '.csv'))
dfs["type"] = "sf"


# concat
df = pd.concat([dfo, dfs], axis=0, ignore_index=True, sort=False)
df[['SF', 'var', 'horizon']] = df['variable_name'].str.split('_', n=2, expand=True)
# keep only forecast
df = df[df["variable_name"].str.contains("SF", na=False, case=False)]
# obs has newer dates
maxdate = df[df["type"] == 'sf']['date'].max()
df = df[df['date'] <= maxdate].copy()
# now keep forcast made in month startMonth
df["date"] = pd.to_datetime(df["date"], errors="coerce")
mask = df["date"].dt.month == startMonth      # Boolean Series
df_month = df.loc[mask].copy()
df_month.to_csv(os.path.join(dir_out, 'test.csv'), index=False)

df_month["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
df_month = df_month.reset_index(drop=True)
mask = df_month["horizon"].between(1, lastHorizon, inclusive="both")   # pandas ≥1.3 uses inclusive="both"
df_filt = df_month.loc[mask].copy()
agg = (
    df_filt
    .groupby(["adm_id", "date", "type", "var"], as_index=False)   # keep keys as columns
    .agg(mean_avg=("mean", "mean"))   # rename the aggregated column to something clear
)
agg.to_csv(os.path.join(dir_out, 'test_agg.csv'), index=False)

def plot(df, fn):
    df["obs"] = pd.to_numeric(df["obs"], errors="coerce")
    df["sf"] = pd.to_numeric(df["sf"], errors="coerce")
    # Drop rows where either value is missing
    df_clean = df.dropna(subset=["obs", "sf"])
    from scipy.stats import pearsonr
    r, p_val = pearsonr(df_clean["obs"], df_clean["sf"])
    # Format the text that will appear on the plot
    corr_text = f"$r$ = {r:.3f}\n$p$ = {p_val:.3g}"
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_clean, x="obs", y="sf", s=60, edgecolor="k", linewidth=0.5, alpha=0.8)
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
    plt.tight_layout()
    plt.savefig(fn, dpi=300, bbox_inches='tight')

obs = agg[(agg["type"] =='obs') & (agg["var"] == 'r')]["mean_avg"].tolist()
sf =  agg[(agg["type"] =='sf') & (agg["var"] == 'r')]["mean_avg"].tolist()
df = pd.DataFrame({'obs': obs, 'sf': sf})
plot(df, os.path.join(dir_out, 'rain_corr.png'))

df['obs'] = (df['obs'] - df['obs'].mean()) / df['obs'].std()
df['sf'] = (df['sf'] - df['sf'].mean()) / df['sf'].std()
plot(df, os.path.join(dir_out, 'z_rain_corr.png'))

obs = agg[(agg["type"] =='obs') & (agg["var"] == 't')]["mean_avg"].tolist()
sf =  agg[(agg["type"] =='sf') & (agg["var"] == 't')]["mean_avg"].tolist()
df = pd.DataFrame({'obs': obs, 'sf': sf})
plot(df, os.path.join(dir_out, 'temp_corr.png'))
print()


