from F_post_processsing import F100_analyze_hindcast_output
from A_config import a10_config


if __name__ == '__main__':
    # USER PARAMS
    metric = 'rRMSE_p' #metric for best model selection
    # config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZAsummer/ZAsummer_config.json'
    # debug
    # config_fn = r'V:\foodsec\Projects\SNYF\ZA_test_new_code\ZAsummer_config.json'
    config_fn = r'V:\foodsec\Projects\SNYF\NDarfur\NDarfur_config.json'


    run_name = 'months56'

    # END OF USER PARAMS

    config = a10_config.read(config_fn, run_name)
    # debug
    # config.models_out_dir = r'U:\data\cid-bulk22\Shared\tmp\projectData\ML4CAST\tmp_buttami\Output'
    # config.models_spec_dir = r'U:\data\cid-bulk22\Shared\tmp\projectData\ML4CAST\tmp_buttami\Specs'
    F100_analyze_hindcast_output.gather_output(config)
    F100_analyze_hindcast_output.compare_outputs(config, metric2use = metric)


