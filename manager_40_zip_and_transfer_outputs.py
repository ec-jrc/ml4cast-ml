from A_config import a10_config
import shutil
import os

if __name__ == '__main__':
    """
    Main function getting outputs once condor has terminated
    Files are zipped and copied here on bdap: /mnt/cidstorage/cidportal/data/cid-bulk22/Shared/tmp/projectData/ML4CAST/
    On offcie pc I can find it here: U:\data\cid-bulk22\Shared\tmp\projectData\ML4CAST
    """
    ##########################################################################################
    # USER PARAMS
    config_fn = r'/eos/jeodpp/data/projects/ML4CAST/ZA/summer/ZAsummer_Maize_(corn)_WC-South_Africa-ASAP_config.json'
    run_name = '20240920_50_maize'
    # END OF USER PARAMS
    ##########################################################################################
    config = a10_config.read(config_fn, run_name, run_type='fast_tuning')  # only to get root dir
    run_dir = config.models_dir

    # config.models_dir is the output dir (specs and out)
    jeo_share_root = '/mnt/cidstorage/cidportal/data/cid-bulk22/Shared/tmp/projectData/ML4CAST/'
    condor_log_root = '/mnt/jeoproc/log/ml4castproc/' + config.AOI + '/'

    # zip and transfer logs
    output_filename = os.path.join(jeo_share_root, 'TUNE_' + run_name + '_logs')
    dir_name = condor_log_root
    shutil.make_archive(output_filename, 'zip', dir_name)

    # zip and transfer outputs
    output_filename = os.path.join(jeo_share_root, 'TUNE_' + run_name) #config.AOI + '_' + run_name + '_model_outputs')
    dir_name = config.models_dir
    shutil.make_archive(output_filename, 'zip', run_dir)
