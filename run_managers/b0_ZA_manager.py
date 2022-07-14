import time
import datetime
import src.constants as cst


# from preprocess import b20_LoadCsv_savePickle
# from preprocess import b30_saveXprctProductionPickle
# from preprocess import b31_map90prctProducer
# from preprocess import b32_trend_analysis
# from preprocess import b40_plots_and_pheno
# from preprocess import b60_build_features
# from preprocess import b65_check_data
# from preprocess import b70_data_exploration
import b100_usetting_and_mod_manager
#from postprocess_analysis import b200_gather_output
#from postprocess_analysis import b215_compare_best_config_outputs_by_model
#from postprocess_analysis import b_230_best_model_scatter
target = 'ZAsummer'



start_time = time.time()
print(datetime.datetime.now())

if (True):
    # b20_LoadCsv_savePickle.LoadCsv_savePickle(target, 'Predictors_ERA5')
    # b30_saveXprctProductionPickle.saveXprctProductionPickle(target)
    # b31_map90prctProducer.map90prctProducer(target)
     #b32_trend_analysis.trend_analysis(target)
    # b40_plots_and_pheno.plots_and_pheno(target)
    # b60_build_features.build_features(target)
    # b65_check_data.check(target)
    # b70_data_exploration.explore(target)
    b100_usetting_and_mod_manager.usetting_and_model_manager(target)

    #dir_run_of_interest = cst.dir_run_of_interest_b200_b215
    #b200_gather_output.gather_output(dir_run_of_interest)
    #b215_compare_best_config_outputs_by_model.compare_outputs(dir_run_of_interest, target)
    #b_230_best_model_scatter.plot_best_scatter(dir_run_of_interest, target)
print("--- %s seconds ---" % (time.time() - start_time))