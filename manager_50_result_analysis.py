from F_post_processsing import F100_analyze_hindcast_output
from A_config import a10_config
import manager_0_user_params as upar


if __name__ == '__main__':
    # USER PARAMS
    metric = upar.metric #metric for best model selection, while rRSD_sor_mil_20250428MSE_p only this implemnted for now because of graphs and maps
    config_fn = upar.config_fn
    run_name = upar.run_name

    config = a10_config.read(config_fn, run_name)
    fn_shape_gaul1 = config.fn_reference_shape #r'V:\foodsec\Projects\SNYF\stable_input_data\SHP_Files\gaul1_asap.shp'
    country_name_in_shp_file = config.country_name_in_shp_file
    # Fetch all run output and copy to all_model_output.csv in dir analysisOutputDir
    F100_analyze_hindcast_output.gather_output(config)
    # Make the analysis and plot
    F100_analyze_hindcast_output.compare_outputs(config, fn_shape_gaul1, country_name_in_shp_file,
                                                 gdf_gaul0_column=config.gaul0_column_name_in_shp_file, metric2use=metric)



