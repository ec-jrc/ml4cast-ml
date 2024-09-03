import sys
from F_post_processsing import F100_analyze_hindcast_output
from A_config import a10_config


if __name__ == '__main__':
    # USER PARAMS
    metric = 'rRMSE_p' #metric for best model selection
    country_name_in_shp_file = 'South Africa'
    # env = 'pc' #['pc','jeo']
    # if env == 'pc':
    if 'win' in sys.platform:
        # config_fn = r'V:\foodsec\Projects\SNYF\NDarfur\NDarfur_config.json'
        # config_fn = r'V:\foodsec\Projects\SNYF\ZA_test_new_code\ZAsummer_config.json'
        # run_name = 'RUN_final_jeo_ndvi_geoterra'
        config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\ZA\summer\ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'  # r'V:\foodsec\Projects\SNYF\NDarfur\NDarfur_config.json'
        run_name = 'months5onlyMaize'#'months5and7'  # 'test_quick'
        fn_shape_gaul1 = r'F:\DATA\ASAP ref data\2023 06\gaul1_asap_v04\gaul1_asap.shp'
    else:
        config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZAsummer/ZAsummer_config.json'
        run_name = ''
        fn_shape_gaul1 = r''
    # END OF USER PARAMS

    config = a10_config.read(config_fn, run_name)
    F100_analyze_hindcast_output.gather_output(config)
    F100_analyze_hindcast_output.compare_outputs(config, fn_shape_gaul1, country_name_in_shp_file,  gdf_gaul0_column='name0', metric2use=metric)



