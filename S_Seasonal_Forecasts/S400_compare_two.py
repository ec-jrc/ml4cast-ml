import pandas as pd
from A_config import a10_config
import os
import seaborn as sns
import matplotlib.pyplot as plt
from F_post_processsing import F100_analyze_hindcast_output
"""
Compare two results (e.g. without SF vs with SF) 
"""

# configuration files and run names
baseDir = r'V:\foodsec\Projects\SNYF\SIDv\TN\_SF_test'
cf1 = os.path.join(baseDir, 'NO_SF_baseline\TNMultiple_WC-Tunisia-ASAP_config.json')
rn1 = 'TNv_NoSF'
short_name1 = 'noSF'
cf2 = os.path.join(baseDir, 'ObsAsSF\TNMultiple_WC-Tunisia-ASAP_config_ObsAsForecast.json')
rn2 = 'TNv_ObsAsSF'
short_name2 = 'ObsAsSF'
########################################################

dir_out = os.path.join(baseDir, 'comp_' + rn1 + '_vs_' + rn2)
os.makedirs(dir_out, exist_ok=True)
mlsettings = a10_config.mlSettings(forecastingMonths=0)
# to gather: list of times, list of crops; a df with:
# run_name, time, crop, model (get all for now), avg rmse_perct, avg r2 by admin, same at national lvel


df = pd.DataFrame()
configs = [a10_config.read(cf1, rn1), a10_config.read(cf2, rn2)]
runs = [rn1, rn2]
short_names = [short_name1, short_name2]
for run, config, short_name in zip(runs, configs, short_names):
    analysisOutputDir = os.path.join(config.models_out_dir, 'Analysis')
    b1 = pd.read_csv(analysisOutputDir + '/' + 'all_model_best1.csv')
    # metric2use = 'rRMSE_p'
    # var4time = 'forecast_time'
    # mlsettings = a10_config.mlSettings(forecastingMonths=0)
    # b1, b4 = F100_analyze_hindcast_output.extract_best_1_and_4(mo, metric2use, var4time, config, mlsettings)
    b1.insert(0, 'run_name', run)
    b1.insert(1, 'run_short_name', short_name)
    df = pd.concat([df, b1], ignore_index=True)
list_time = df.forecast_time.unique()
list_crop = df.Crop.unique()
for crop in list_crop:
    #plot rrmse of ML and bench trough time (ML and Tab has two versions, run1 and run2)
    fig, axs = plt.subplots(nrows=2, ncols=1)
    # get max
    ymax = df[df['Crop'] == crop]['rRMSE_p'].max()
    df2plot = df[(df['run_short_name']==short_name1) & (df['Crop']==crop)]
    df2plot['Estimator'] = df2plot['Estimator'].map(lambda x: x if x in mlsettings.benchmarks else 'ML')
    p = sns.barplot(df2plot, x="forecast_time", y="rRMSE_p", hue="Estimator", ax=axs[0])
    plt.ylim(0, ymax)
    p.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[0].set_title(short_name1)

    df2plot = df[(df['run_short_name'] == short_name2) & (df['Crop'] == crop)]
    df2plot['Estimator'] = df2plot['Estimator'].map(lambda x: x if x in mlsettings.benchmarks else 'ML')
    p = sns.barplot(df2plot, x="forecast_time", y="rRMSE_p", hue="Estimator", ax=axs[1])
    plt.ylim(0, ymax)
    p.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[1].set_title(short_name2)
    fig_name = os.path.join(dir_out, crop +'.png')
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()

# same but reshaped in one graph
# check that  ["Null_model", "Trend", "PeakNDVI"] have same "rRMSE_p" at each 'forecast_time', no matter 'run_short_name'
for crop in list_crop:
    for t in df.forecast_time.unique():
        for est in ["Null_model", "Trend", "PeakNDVI"]:
            tmp = df[(df['Crop'] == crop) & (df['forecast_time'] == t) & (df['Estimator'] == est)]
            if tmp["rRMSE_p"].nunique() != 1:
                print(crop, t, est)
                print("Not all rRMSE_p values are equal")

print()


