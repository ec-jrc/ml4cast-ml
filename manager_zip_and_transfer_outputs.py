from A_config import a10_config
import shutil
import os

if __name__ == '__main__':
    """
    Main function getting outputs once condor has terminated
    """
    ##########################################################################################
    # USER PARAMS
    config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZAsummer/ZAsummer_config.json'
    run_name = 'month5and8'
    # END OF USER PARAMS
    ##########################################################################################

    config = a10_config.read(config_fn, run_name)
    # config.models_dir is the output dir (specs and out)
    jeo_share_root = '/mnt/cidstorage/cidportal/data/cid-bulk22/Shared/tmp/projectData/ML4CAST/'
    condor_log_root = '/mnt/jeoproc/log/ml4castproc/' + config.AOI + '/'

    # zip and transfer logs
    # output_filename = os.path.join(jeo_share_root, config.AOI + '_logs')
    # dir_name = condor_log_root
    # shutil.make_archive(output_filename, 'zip', dir_name)

    # zip and transfer outputs
    output_filename = os.path.join(jeo_share_root, config.AOI + '_model_outputs')
    dir_name = config.models_dir
    dir_name = "/eos/jeodpp/data/projects/ML4CAST/ZAsummer/MLYF/RUN_month5and8_TUNING/"
    shutil.make_archive(output_filename, 'zip', dir_name)
