import pandas as pd
import os

# This function is placed in a separate python file and does not require matplotlib in tuning (it was slowing down condor)

def LoadCleanedLabel(config):
    return pd.read_csv(os.path.join(config.data_dir, config.AOI + '_STATS_cleaned' + str(config.prct2retain)+'.csv'))