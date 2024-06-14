import sys
import json
import time
import os
from A_config import a10_config
from D_modelling import d090_model_wrapper

def launcher(fn, config_fn, run_name):
    config = a10_config.read(config_fn, run_name)
    d090_model_wrapper.fit_and_validate_single_model(fn, config, 'tuning')

if __name__ == '__main__':
    #print(sys.version)
    uset_file = r'{}'.format(sys.argv[1])
    config_fn = r'{}'.format(sys.argv[2])
    run_name =  r'{}'.format(sys.argv[3])
    launcher(uset_file, config_fn, run_name)

