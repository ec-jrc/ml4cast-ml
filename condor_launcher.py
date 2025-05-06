import sys
import os
from A_config import a10_config
from D_modelling import d090_model_wrapper


def launcher(fn, config_fn, run_name, runType):
    # limit threads for Condor
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    os.environ['OMP_NUM_THREADS'] = '2'
    config = a10_config.read(config_fn, run_name, run_type=runType)

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
