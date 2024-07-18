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
    # END OF USER PARAMS
    ##########################################################################################
    config = a10_config.read(config_fn, 'dummy_name')
    # config.models_dir is the output dir (specs and out)
    jeo_share_root = '/mnt/cidstorage/cidportal/data/cid-bulk22/Shared/tmp/projectData/ML4CAST/'
    condor_log_root = '/mnt/jeoproc/log/ml4castproc/' + config.AOI + '/'
    # zip and transfer logs
    output_filename = os.path,jeo_share_root(jeo_share_root, config.AOI + '_logs.zip')
    dir_name = condor_log_root
    shutil.make_archive(output_filename, 'zip', dir_name)
    # zip and transfer outputs
    output_filename = os.path, jeo_share_root(jeo_share_root, config.AOI + '_model_outputs.zip')
    dir_name = config.models_dir
    shutil.make_archive(output_filename, 'zip', dir_name)

