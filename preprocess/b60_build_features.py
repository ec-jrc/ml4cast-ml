import pandas as pd
import numpy as np
import b05_Init
from preprocess import f_dek_utilities
import datetime
from dateutil.relativedelta import *

def build_features(target, ope_run=False):
    '''Feature by pheno periods:
    - EXPANSION: SOS-TOM
    - MATURATION: TOM-SEN
    - SENESCENCE: SEN-EOS'''

    pd.set_option('display.max_columns', None)
    desired_width = 320
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns', 10)

    project = b05_Init.init(target)
    #get the avg pheno
    dirPheno = project['output_dir'] + '/Pheno'
    dirOut = project['output_dir']
    if ope_run:
        dirOut = dirOut + '/OPE_RUN'
    pheno = pd.read_csv(dirPheno + '/' +project['AOI'] + '_pheno_stats.csv')#_season1.csv')
    vals = pheno.loc[pheno['Stats'] == 'mean', ['SOS', 'TOM', 'SEN', 'EOS']].values.flatten()
    sos_tom_sen_eos_DEK = np.round(vals).astype(int)
    #define month and day of each pheno timing
    f = lambda x: f_dek_utilities.f_dek_year2dateFirstDekDay(np.asscalar(x), 2000)
    sos_tom_sen_eos_DateOf2000 = [f(x) for x in sos_tom_sen_eos_DEK]

    #reorder years
    sos_tom_sen_eos_dYear = np.zeros(4).astype(int)
    # Find timing that are later than EOS, and must go the year before
    sub = [i for i, x in enumerate(sos_tom_sen_eos_DateOf2000) if x > sos_tom_sen_eos_DateOf2000[3]]
    sos_tom_sen_eos_dYear[sub] = -1
    #f = lambda x: x.month
    sos_tom_sen_eos_Month = [x.month for x in sos_tom_sen_eos_DateOf2000]
    sos_tom_sen_eos_Day = [x.day for x in sos_tom_sen_eos_DateOf2000]
    # now adjust the year in sos_tom_sen_eos_DateOf2000
    sos_tom_sen_eos_DateOf2000 = [datetime.date(x.year + y, x.month, x.day) for x, y in zip(sos_tom_sen_eos_DateOf2000, sos_tom_sen_eos_dYear)]
    # open all variables and make composites by pheno phase
    x = pd.read_pickle(dirOut + '/' + project['AOI'] + '_predictors.pkl')
    x['DateIndex'] = x['Date']
    x = x.set_index('DateIndex')
    # assign pheno phase to each period
    x['Pheno_phase'] = 'None'
    x['Pheno_phaseID'] = ''
    x['YearOfEOS'] = ''
    # assign month to each period (we consider the full first month and the full last month coming from pheno
    x['Month'] = ''
    first_month = sos_tom_sen_eos_Month[0]
    last_month = sos_tom_sen_eos_Month[-1]
    d1 = datetime.date(2000 + sos_tom_sen_eos_dYear[-1], sos_tom_sen_eos_Month[-1], 1)
    d2 = datetime.date(2000 + sos_tom_sen_eos_dYear[0], sos_tom_sen_eos_Month[0], 1)
    n_months = (d1.year - d2.year) * 12 + d1.month - d2.month + 1
    # save dates info
    tmp = pd.DataFrame(sos_tom_sen_eos_DateOf2000, columns=['Start_stage'])
    tmp['Stage'] = ['sos', 'tom', 'sen', 'eos']
    tmp['Stage_ID'] = [0,1,2,3]
    tmp['Month'] = sos_tom_sen_eos_Month
    def month2ID(ms):
        msID = ms.copy()
        msID[0] = 1 #assign 1 to the first month
        for i in range(1,len(ms)):
            if ms[i] >= ms[0]:
                msID[i] = ms[i] - ms[0] + 1
            else:
                msID[i] = ms[i] + 12 - ms[0] + 1
        return msID
    tmp['Month_ID'] = month2ID([x.month for x in sos_tom_sen_eos_DateOf2000])
    tmp.to_csv(dirOut + '/' + project['AOI'] + '_pheno_mean_used.csv')

    minYear = x.index.min().year
    maxYear = x.index.max().year
    for i in range(minYear, maxYear+1, 1):
        #sos to tom
        start_date = str(i+sos_tom_sen_eos_dYear[0])+'-'+str(sos_tom_sen_eos_Month[0])+'-'+str(sos_tom_sen_eos_Day[0])
        end_date = str(i+sos_tom_sen_eos_dYear[1])+'-'+str(sos_tom_sen_eos_Month[1])+'-'+str(sos_tom_sen_eos_Day[1])
        #the date may not be present and I don't want incomplete periods
        if x.first_valid_index() <= datetime.datetime.strptime(start_date, '%Y-%m-%d'):
            x.loc[start_date : end_date,'Pheno_phase'] = 'SOS-TOM'
            x.loc[start_date: end_date, 'Pheno_phaseID'] = 1
            x.loc[start_date: end_date, 'YearOfEOS'] = i + sos_tom_sen_eos_dYear[3]
            #tom to sen
            start_date = str(i + sos_tom_sen_eos_dYear[1]) + '-' + str(sos_tom_sen_eos_Month[1]) + '-' + str(sos_tom_sen_eos_Day[1]+1) #+1. pandas loc is inclusive on both sides
            end_date =   str(i+sos_tom_sen_eos_dYear[2]) + '-' + str(sos_tom_sen_eos_Month[2]) + '-' + str(sos_tom_sen_eos_Day[2])
            x.loc[start_date: end_date, 'Pheno_phase'] = 'TOM-SEN'
            x.loc[start_date: end_date, 'Pheno_phaseID'] = 2
            x.loc[start_date: end_date, 'YearOfEOS'] = i + sos_tom_sen_eos_dYear[3]
            #sen2eos
            start_date = str(i + sos_tom_sen_eos_dYear[2]) + '-' + str(sos_tom_sen_eos_Month[2]) + '-' + str(sos_tom_sen_eos_Day[2] + 1)  # +1. pandas loc is inclusive on both sides
            end_date = str(i + sos_tom_sen_eos_dYear[3]) + '-' + str(sos_tom_sen_eos_Month[3]) + '-' + str(sos_tom_sen_eos_Day[3])
            x.loc[start_date: end_date, 'Pheno_phase'] = 'SEN-EOS'
            x.loc[start_date: end_date, 'Pheno_phaseID'] = 3
            x.loc[start_date: end_date, 'YearOfEOS'] = i + sos_tom_sen_eos_dYear[3]
            # now fill all months
            for m in range(0, n_months):
                start_date = datetime.date(i + sos_tom_sen_eos_dYear[0], sos_tom_sen_eos_Month[0], 1)
                start_date = start_date + relativedelta(months=+m)
                # go to the next month and back for one day to get the last day of the month
                end_date = start_date + relativedelta(months=+1) + relativedelta(days=-1)
                x.loc[start_date.strftime('%Y-%m-%d'): end_date.strftime('%Y-%m-%d'), 'Month'] = m + 1
                x.loc[start_date.strftime('%Y-%m-%d'): end_date.strftime('%Y-%m-%d'), 'YearOfEOS'] = i + sos_tom_sen_eos_dYear[3]
    # Extract the pheno features
    y = x.groupby(by=[x['variable_name'], x['AU_code'], x['Pheno_phase'], x['YearOfEOS']]). \
        agg(Pheno_phaseID=('Pheno_phaseID','first'), AU_name=('AU_name', 'first'), Au_code1=('AU_code', 'first'), ASAP1_ID=('ASAP1_ID', 'first'),
            Date=('Date', 'first'),mean=('mean', 'mean'),
            min=('mean', 'min'), max=('mean', 'max'), sum=('mean', 'sum'))
    y = y.reset_index()
    y = y.drop(y[y['Pheno_phase'] == 'None'].index)
    # Extract the monthly features
    k = x.groupby(by=[x['variable_name'], x['AU_code'], x['Month'], x['YearOfEOS']]). \
        agg(AU_name=('AU_name', 'first'), Au_code1=('AU_code', 'first'),
            ASAP1_ID=('ASAP1_ID', 'first'),
            Date=('Date', 'first'), mean=('mean', 'mean'),  # Year=('Year','first'),Month=('Month','first'),
            min=('mean', 'min'), max=('mean', 'max'), sum=('mean', 'sum'))
    k = k.reset_index()
    k = k.drop(k[k['Month'] == ''].index)

    timerange = project['timeRange']
    y = y.drop(y[y['YearOfEOS'] < timerange[0]].index)
    y.to_csv(dirOut + '/' + project['AOI'] +'_pheno_features.csv')
    y.to_pickle(dirOut + '/' + project['AOI'] +'_pheno_features.pkl')
    k = k.drop(k[k['YearOfEOS'] < timerange[0]].index)
    k.to_csv(dirOut + '/' + project['AOI'] + '_monthly_features.csv')
    k.to_pickle(dirOut + '/' + project['AOI'] + '_monthly_features.pkl')

    #Reshape the df to scikit format
    PhenoSep = 'P'
    MonthSep = 'M'
    ivars = project['ivars']
    ivars_short = project['ivars_short']
    AU_list = y['AU_code'].unique()
    init = 0
    for au in AU_list:
        df_au = y[y['AU_code'] == au]
        dfM_au = k[k['AU_code'] == au]
        YY_list = df_au['YearOfEOS'].unique()
        for yy in YY_list:
            df_au_yy =  df_au[df_au['YearOfEOS'] == yy]
            dfM_au_yy = dfM_au[dfM_au['YearOfEOS'] == yy]
            row = [au,df_au['ASAP1_ID'].iloc[0],df_au['AU_name'].iloc[0], df_au_yy['YearOfEOS'].iloc[0]]
            #rowM = [au, dfM_au['ASAP1_ID'].iloc[0], dfM_au['AU_name'].iloc[0], dfM_au_yy['YearOfEOS'].iloc[0]]
            columns = ['AU_code', 'ASAP1_ID', 'AU_name', 'YearOfEOS']
            for v, vs in zip(ivars, ivars_short):
                df_au_yy_v = df_au_yy[df_au_yy['variable_name'] == v]
                dfM_au_yy_v = dfM_au_yy[dfM_au_yy['variable_name'] == v]
                pp_list = np.sort(df_au_yy_v['Pheno_phaseID'].unique())
                for pp in pp_list:
                    # now is one variable during a pheno phase of a year of an AU (all info in the list)
                    df_au_yy_v_pp = df_au_yy_v[df_au_yy_v['Pheno_phaseID'] == pp]
                    if v == 'rainfall':
                        row.append(df_au_yy_v_pp['sum'].iloc[0])
                        columns.append(vs + 'Sum' + PhenoSep + str(pp))
                    else:
                        row.append(df_au_yy_v_pp['mean'].iloc[0])
                        columns.append(vs + PhenoSep + str(pp))
                    if (v == 'temperature') or (v == 'NDVI'):
                        columns.append(vs + 'min' + PhenoSep + str(pp))
                        row.append(df_au_yy_v_pp['min'].iloc[0])
                        columns.append(vs + 'max' + PhenoSep + str(pp))
                        row.append(df_au_yy_v_pp['max'].iloc[0])
                mm_list = np.sort(dfM_au_yy_v['Month'].unique())
                for mm in mm_list:
                    # now is one variable during a pheno phase of a year of an AU (all info in the list)
                    dfM_au_yy_v_pp = dfM_au_yy_v[dfM_au_yy_v['Month'] == mm]
                    if v == 'rainfall':
                        row.append(dfM_au_yy_v_pp['sum'].iloc[0])
                        columns.append(vs + 'Sum' + MonthSep + str(mm))
                    else:
                        row.append(dfM_au_yy_v_pp['mean'].iloc[0])
                        columns.append(vs + MonthSep + str(mm))
                    if (v == 'temperature') or (v == 'NDVI'):
                        columns.append(vs + 'min' + MonthSep + str(mm))
                        row.append(dfM_au_yy_v_pp['min'].iloc[0])
                        columns.append(vs + 'max' + MonthSep + str(mm))
                        row.append(dfM_au_yy_v_pp['max'].iloc[0])
            if init == 0:
                df = pd.DataFrame([row], columns=columns)
                init = 1
            else:
                df2 = pd.DataFrame([row], columns=columns)
                df = pd.concat([df, df2])
    df.to_csv(dirOut + '/' + project['AOI'] + '_pheno_features4scikit.csv')
    df.to_pickle(dirOut + '/' + project['AOI'] + '_pheno_features4scikit.pkl')



