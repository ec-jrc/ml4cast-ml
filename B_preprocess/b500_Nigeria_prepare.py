import pandas as pd
import numpy as np
import os
import shutil

"""
Split Nigeria (Wet, sorghum and maize) in two items: mono and bimodal areas. Prepare Tuning data. Config must be adapted manually
"""
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

root = r'V:\foodsec\Projects\SNYF\stable_input_data\NG'
original_dir = 'Wet'

modality_fn = r'lt_stats_maize_for_pheno_by_admin.xlsx'
df = pd.read_excel(os.path.join(root, modality_fn))
modality = ['bi', 'mono']
for m in modality:
    df_mod = df[df['seasonality'] == m]
    print('AVG start and stop to be used in config')
    print('Modality: ' + m, df_mod.START1.mean(), df_mod.STOP1.mean())
    # make dirs
    #Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    new_dir = os.path.join(root, original_dir + '_' + m)
    os.makedirs(new_dir, exist_ok=True)
    shutil.copy(os.path.join(root, original_dir, 'NGMaize_(corn)_WC-HARVESTAT_config.json'),
                os.path.join(new_dir, 'NGMaize_(corn)_WC-HARVESTAT_config.json'))
    new_tuning_dir = os.path.join(new_dir, 'Tuning_data')
    original_tuning_dir = os.path.join(root, original_dir, 'Tuning_data')
    os.makedirs(new_tuning_dir, exist_ok=True)


    # copy files
    prefix = 'NGWet_'
    # files_to_copy = [f for f in os.listdir(original_tuning_dir) if f.startswith(prefix)]
    #files_to_copy = ['NGWet_CROP_id.csv', 'NGWet_measurement_units.csv', 'NGWet_REGION_id.csv', 'NGWet_STATS_cleaned90.csv']
    files_to_copy = ['NGWet_CROP_id.csv', 'NGWet_measurement_units.csv', 'NGWet_REGION_id.csv', 'NGWet_STATS.csv']
    for i, file in enumerate(files_to_copy):
        source_path = os.path.join(original_tuning_dir, file)
        filename, file_extension = os.path.splitext(file)
        extracted_string = filename.split(prefix)[1]
        dest_path = os.path.join(new_tuning_dir, f"{prefix}{m}_{extracted_string}{file_extension}")
        try:
            shutil.copy(source_path, dest_path)
            # print(f"File '{file}' copied to '{dest_path}' successfully.")
        except Exception as error:
            print(f"Error copying file: {error}")
    # copy aspa extraction
    fn = 'Maize_(corn)_WC-Nigeria-HARVESTAT.csv'
    shutil.copy(os.path.join(original_tuning_dir, fn), os.path.join(new_tuning_dir, fn))

    # work on 90% stats, get stats results and drop those not in the modality
    dfSTATS = pd.read_csv(dest_path)
    listAdmibWithMod_m = df_mod['adm_name|first'].unique().tolist()
    dfSTATS = dfSTATS.loc[dfSTATS['adm_name'].isin(listAdmibWithMod_m), :]
    dfSTATS.to_csv(dest_path, index=False)
print('Adjust the config!!!!!!!!!!')