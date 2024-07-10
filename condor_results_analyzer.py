from F_post_processsing import F100_analyze_hindcast_output
from A_config import a10_config


if __name__ == '__main__':
    # USER PARAMS
    metric = 'rRMSE_p' #metric for best model selection
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZAsummer/ZAsummer_config.json'
    # debug
    config_fn = r'V:\foodsec\Projects\SNYF\ZA_test_new_code\ZAsummer_config.json'


    run_name = 'month5'
    forecastingMonths = [5]
    # END OF USER PARAMS

    config = a10_config.read(config_fn, run_name)
    # debug
    config.models_out_dir = r'U:\data\cid-bulk22\Shared\tmp\projectData\ML4CAST\tmp_buttami\Output'
    F100_analyze_hindcast_output.gather_output(config.models_out_dir)
    F100_analyze_hindcast_output.compare_outputs(config.models_out_dir, config, metric2use = metric)



