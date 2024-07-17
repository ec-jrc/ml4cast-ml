from A_config import a10_config
from B_preprocess import b50_yield_data_analysis
from E_viz import e50_yield_data_analysis

if __name__ == '__main__':
    """
    Main function for analysing yield data
     
    """
    ##########################################################################################
    # USER PARAMS
    env = 'pc'  # ['pc','jeo']
    if env == 'pc':
        fn_shape_gaul1 = r'F:\DATA\ASAP ref data\2023 06\gaul1_asap_v04\gaul1_asap.shp'
        config_fn = r'V:\foodsec\Projects\SNYF\ZA_test_new_code\ZAsummer_config.json'
    else:
        fn_shape_gaul1 = ''
        config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZAsummer/ZAsummer_config.json'
    # END OF USER PARAMS
    ##########################################################################################
    config = a10_config.read(config_fn, 'dummy_name')
    country_name_in_shp_file = 'South Africa'
    prct2retain = 100
    b50_yield_data_analysis.saveYieldStats(config, prct2retain=prct2retain)
    e50_yield_data_analysis.mapYieldStats(config, fn_shape_gaul1, country_name_in_shp_file, gdf_gaul0_column='name0', prct2retain=prct2retain)