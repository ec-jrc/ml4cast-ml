import pandas as pd

df = pd.read_csv(r'V:\foodsec\Projects\SNYF\SIDv\TN\SF\NO_SF_baseline\Tuning_data\Multiple_WC-Tunisia-ASAP.csv')

# df = pd.read_csv(r'V:\foodsec\Projects\SNYF\SIDv\TN\SF\SF\Tuning_data\TN_WC_ecmwf_extraction_results_full_archive_e_means.csv')
df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
df.to_csv(r'V:\foodsec\Projects\SNYF\SIDv\TN\SF\SF\Tuning_data\pippo.csv', index=False)
print()

# df2 = pd.read_csv(r'V:\foodsec\Projects\SNYF\SIDv\TN\SF\NO_SF_baseline\Tuning_data\Multiple_WC-Tunisia-ASAP.csv')
