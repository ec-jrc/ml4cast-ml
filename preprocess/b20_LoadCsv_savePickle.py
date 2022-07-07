import pandas as pd
import numpy as np
from preprocess import f_dek_utilities
import b05_Init
from pathlib import Path

def LoadCsv_savePickle(target, predictors_dir):
    ''' Import ASAP data in the csv format, save csv and pickle '''

    pd.set_option('display.max_columns', None)
    desired_width=320
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns',10)

    project = b05_Init.init(target)
    #dirPredictors = project['input_dir']+ '/Predictors_ERA5'
    dirPredictors = project['input_dir'] + '/' + predictors_dir
    dirStat = project['input_dir'] + '/CountryStats'
    dirOut =  project['output_dir']

    # General part
    # read the table with id and wilaya name
    regNames = pd.read_csv(dirStat + '/' + project['AOI'] + '_REGION_id.csv')
    # deal with output directory
    dirOut = Path(dirOut)
    dirOut.mkdir(parents=True, exist_ok=True)

    # read all the ASAP data at once
    x = pd.read_csv(dirPredictors + '/' + project['AOI'] + '_ASAP_data.csv')
    x = x[x['class_name']=='crop']
    x = x[x['classset_name'] == 'static masks']
    # now link it with AU_code and name
    x = pd.merge(x, regNames, left_on=['reg0_id'], right_on=['ASAP1_ID'])
    # get first NDVI time and drop everything before
    minDate = x[x['variable_name'] == 'NDVI']['date'].min()
    x = x[x['date'] >= minDate]
    # fix date, add a column with date
    x['Date'] = pd.to_datetime(x['date'], format='%Y-%m-%d')
    # remove useless stuff
    x = x.drop(['var_id','classset_name','classesset_id','class_name','class_id','date'], axis=1)
    # add dekad of the year
    x['dek'] = x['Date'].map(f_dek_utilities.f_datetime2dek)
    # save files
    fn = project['AOI'] + '_predictors.pkl'
    x.to_pickle(dirOut / fn)
    fn = project['AOI'] +'_predictors.csv'
    x.to_csv(dirOut / fn)

#res = LoadCsv_savePickle('Algeria',  'Predictors_ERA5')