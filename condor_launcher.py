import sys
from A_config import a10_config
from D_modelling import d090_model_wrapper

def launcher(fn, config_fn, run_name, runType):
    config = a10_config.read(config_fn, run_name, run_type=runType)
    d090_model_wrapper.fit_and_validate_single_model(fn, config, runType)



if __name__ == '__main__':
    #print('Arrived in condor_launcher')
    uset_file = r'{}'.format(sys.argv[1])
    config_fn = r'{}'.format(sys.argv[2])
    run_name =  r'{}'.format(sys.argv[3])
    runType = r'{}'.format(sys.argv[4])
    launcher(uset_file, config_fn, run_name, runType)
#
