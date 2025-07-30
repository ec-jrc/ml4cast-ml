import sys
import os
import shutil
from A_config import a10_config
from D_modelling import d090_model_wrapper


def launcher(fn, config_fn, run_name, runType):
    # limit threads for Condor
    modelSettings = a10_config.mlSettings(forecastingMonths=0)
    os.environ['MKL_NUM_THREADS'] = str(modelSettings.condor_param['NUM_THREADS']) #'2'
    os.environ['OPENBLAS_NUM_THREADS'] = str(modelSettings.condor_param['NUM_THREADS']) #'2'
    os.environ['OMP_NUM_THREADS'] = str(modelSettings.condor_param['NUM_THREADS']) #'2'
    os.environ["TABPFN_MODEL_CACHE_DIR"] = "/scratch2/ML4CAST/"
    os.environ["http_proxy"] = "http://proxy-htcondor.cidsn.jrc.it:8888;https_proxy=http://proxy-htcondor.cidsn.jrc.it:8888"
    os.environ["https_proxy"] = "http://proxy-htcondor.cidsn.jrc.it:8888;https_proxy=http://proxy-htcondor.cidsn.jrc.it:8888"
    config = a10_config.read(config_fn, run_name, run_type=runType)
    source_path = os.path.join(config.root_dir, "tabpfn-v2-regressor.ckpt")
    destination_path = "/scratch2/ML4CAST/"
    shutil.copy(source_path, destination_path)

    if 'win' in sys.platform:
        pass
    else:
        print('Environment variables:')
        for k, v in os.environ.items():
            print(f'{k}={v}')

    d090_model_wrapper.fit_and_validate_single_model(fn, config, runType)



if __name__ == '__main__':
    #print('Arrived in condor_launcher')
    uset_file = r'{}'.format(sys.argv[1])
    config_fn = r'{}'.format(sys.argv[2])
    run_name =  r'{}'.format(sys.argv[3])
    runType = r'{}'.format(sys.argv[4])
    launcher(uset_file, config_fn, run_name, runType)
#
