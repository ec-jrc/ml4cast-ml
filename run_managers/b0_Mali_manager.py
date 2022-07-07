import time
import datetime

from postprocess_analysis import b215_compare_best_config_outputs_by_model

target = 'Mali'

start_time = time.time()
print(datetime.datetime.now())

if (True):
    #b20_LoadCsv_savePickle.LoadCsv_savePickle(target, 'Predictors_ERA5')
    #b30_saveXprctProductionPickle.saveXprctProductionPickle(target)
    #b32_trend_analysis.trend_analysis(target)
    #b31_map90prctProducer.map90prctProducer(target)
    #b40_plots_and_pheno.plots_and_pheno(target)
    #b60_build_features.build_features(target)
    #b65_check_data.check(target)
    #b70_data_exploration.explore(target)
    #b100_usetting_and_mod_manager_revised.usetting_and_model_manager('Mali')
    #b200_gather_output.gather_output('X:/PY_data/ML1_data_output/Mali/Model/JEO_runs/20210521')
    b215_compare_best_config_outputs_by_model.compare_outputs('X:/PY_data/ML1_data_output/Mali/Model/JEO_runs/20210521', 'Mali')
print("--- %s seconds ---" % (time.time() - start_time))