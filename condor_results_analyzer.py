from F_post_processsing import F100_analyze_hindcast_output
from A_config import a10_config


if __name__ == '__main__':
    # USER PARAMS
    run_name = 'month5'
    config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZAsummer/ZAsummer_config.json'
    forecastingMonths = [5]
    # END OF USER PARAMS
    config = a10_config.read(config_fn, run_name)
    F100_analyze_hindcast_output.gather_output(config.models_out_dir)
    F100_analyze_hindcast_output.compare_outputs(config.models_out_dir)
