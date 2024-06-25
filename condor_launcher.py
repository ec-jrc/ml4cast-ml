
# IMPORT TO BE USED in 120
# from mrmr_loc.pandas import mrmr_regression
# print(mrmr_regression)

# from mrmr_loc import bigquery #OK
# print(bigquery)

# from mrmr_loc.main import mrmr_base #THIS GIVE ERROR
# print(mrmr_base)

#
# from tqdm import tqdm #THESE NOT
# print(tqdm)
# import warnings
# warnings.filterwarnings("ignore")
# FLOOR = .001
# print(warnings)
#
import sys
from A_config import a10_config
from D_modelling import d090_model_wrapper

def launcher(fn, config_fn, run_name):
    config = a10_config.read(config_fn, run_name)
    d090_model_wrapper.fit_and_validate_single_model(fn, config, runType="tuning")


if __name__ == '__main__':
    print('Arrived in condor_launcher')
    uset_file = r'{}'.format(sys.argv[1])
    config_fn = r'{}'.format(sys.argv[2])
    run_name =  r'{}'.format(sys.argv[3])
    launcher(uset_file, config_fn, run_name)
#
