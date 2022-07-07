import time
import datetime
#import b50_simple_model
import b100_usetting_and_mod_manager

#import b210_compare_outputs

target = 'Algeria'
start_time = time.time()
print(datetime.datetime.now())

print('debug')

#When adding new adata from ASAP
if (False):
    #b20_LoadCsv_savePickle.LoadCsv_savePickle(target)
    b60_build_features.build_features(target)
    print('debug')

if (False):
    b20_LoadCsv_savePickle.LoadCsv_savePickle(target, 'Predictors_ERA5')
    b30_saveXprctProductionPickle.saveXprctProductionPickle(target)
    b31_map90prctProducer.map90prctProducer(target)
    b40_plots_and_pheno.plots_and_pheno(target)
    #b50_simple_model.simple_model(target)
    b60_build_features.build_features(target)
    #b60_build_features.build_monthly_features(target)
    b70_data_exploration.explore(target)
    b80_cluster_AUs.cluster(target)
    b100_usetting_and_mod_manager.usetting_and_model_manager(target)



print("--- %s seconds ---" % (time.time() - start_time))


