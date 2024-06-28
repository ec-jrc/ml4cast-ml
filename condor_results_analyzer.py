from F_post_processsing import F100_gather_hindcast_output
from A_config import a10_config


if __name__ == '__main__':
    run_name = 'month5'
    config = a10_config.read(config_fn, run_name)
    F100_gather_hindcast_output.gather_output(config.models_out_dir)
