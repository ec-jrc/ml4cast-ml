from A_config import a10_config
from B_preprocess import b50_yield_data_analysis
from E_viz import e50_yield_data_analysis
import sys

if __name__ == '__main__':
    """
    Main function for analysing yield data
     
    """
    ##########################################################################################
    # USER PARAMS
    prct2retain = 100
    # country_name_in_shp_file = 'Zambia' #'South Africa' #'Benin' #'South Africa'
    # env = 'pc'  # ['pc','jeo']
    # if env == 'pc':
    if 'win' in sys.platform:
        # fn_shape_gaul1 = r'F:\DATA\ASAP ref data\2023 06\gaul1_asap_v04\gaul1_asap.shp'
        #config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\BE\main\Maize_(corn)_WC-Benin-ASAP.json'
        #config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\ZA\summer\ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'
        config_fn = r'V:\foodsec\Projects\SNYF\stable_input_data\ZM\annual\Maize_(corn)_WC-Zambia-FEWSNET.json'
    else:
        # fn_shape_gaul1 = ''
        config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZAsummer/ZAsummer_config.json'
    # END OF USER PARAMS
    ##########################################################################################
    config = a10_config.read(config_fn, 'dummy_name')



    adminID_column_name_in_shp_file = config.adminID_column_name_in_shp_file

    b50_yield_data_analysis.saveYieldStats(config, prct2retain=prct2retain)
    e50_yield_data_analysis.mapYieldStats(config, config.fn_reference_shape, config.country_name_in_shp_file, \
                                          gdf_gaul0_column=config.gaul0_column_name_in_shp_file, adminID_column_name_in_shp_file = config.adminID_column_name_in_shp_file, prct2retain=prct2retain)
    e50_yield_data_analysis.trend_anlysis(config)