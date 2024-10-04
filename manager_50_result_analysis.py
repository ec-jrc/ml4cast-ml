import sys
from F_post_processsing import F100_analyze_hindcast_output
from A_config import a10_config


if __name__ == '__main__':
    # USER PARAMS
    metric = 'rRMSE_p' #metric for best model selection
    # env = 'pc' #['pc','jeo']
    # if env == 'pc':
    if 'win' in sys.platform:
        # config_fn = r'V:\foodsec\Projects\SNYF\NDarfur\NDarfur_config.json'
        # config_fn = r'V:\foodsec\Projects\SNYF\ZA_test_new_code\ZAsummer_config.json'
        # run_name = 'RUN_final_jeo_ndvi_geoterra'
        config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\ZA\summer\ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'  # r'V:\foodsec\Projects\SNYF\NDarfur\NDarfur_config.json'
        run_name = '20240924_75_100_maize_sunflower_soybeans'#'20240920_50_maize'
    else:
        config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZAsummer/ZAsummer_config.json'
        run_name = ''
    # END OF USER PARAMS

    config = a10_config.read(config_fn, run_name)
    fn_shape_gaul1 = config.fn_reference_shape #r'V:\foodsec\Projects\SNYF\stable_input_data\SHP_Files\gaul1_asap.shp'
    country_name_in_shp_file = config.country_name_in_shp_file
    F100_analyze_hindcast_output.gather_output(config)
    F100_analyze_hindcast_output.compare_outputs(config, fn_shape_gaul1, country_name_in_shp_file,  gdf_gaul0_column='name0', metric2use=metric)



