import manager_0_user_params as upar
from A_config import a10_config
from B_preprocess import b50_yield_data_analysis
from E_viz import e50_yield_data_analysis



if __name__ == '__main__':
    """
    Main function for analysing yield data
     
    """
    ##########################################################################################
    # USER PARAMS
    prct2retain = upar.prct2retain
    config_fn = upar.config_fn
    ##########################################################################################

    config = a10_config.read(config_fn, 'dummy_name')
    adminID_column_name_in_shp_file = config.adminID_column_name_in_shp_file
    b50_yield_data_analysis.saveYieldStats(config, prct2retain=prct2retain)
    e50_yield_data_analysis.mapYieldStats(config, config.fn_reference_shape, config.country_name_in_shp_file,\
                                          gdf_gaul0_column=config.gaul0_column_name_in_shp_file, adminID_column_name_in_shp_file = config.adminID_column_name_in_shp_file, prct2retain=prct2retain)
    e50_yield_data_analysis.trend_anlysis(config)