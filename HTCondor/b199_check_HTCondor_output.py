import glob
import pandas as pd
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

def check_outputs(dir_output, full_path_to_task_argument):#, pkl_path):
    # check if some output is missing comparing task_arguments.txt and model outputs
    # if it is the case make  a new task_arguments1.txt with only missing file
    tmp = glob.glob(dir_output + '/*_output.csv')
    outputList = [os.path.basename(x) for x in tmp]
    outDF = pd.DataFrame(outputList, columns=['fn'])
    outDF['ID_model'] = outDF['fn'].str[3:18]

    # with open(full_path_to_task_argument) as f:
    #     lines = [line.rstrip() for line in f]
    # taskDF = pd.DataFrame(lines, columns=['task'])
    taskDF = pd.read_csv(full_path_to_task_argument, delim_whitespace=True, header=None)
    taskDF.rename(columns={0: 'ID_task', 1: 'pickle_path'}, inplace=True)
    #make a copy and keep it untouched
    taskDFc = taskDF.copy()

    # taskDF['ID_task'] = taskDF['task'].str[0:15]
    # taskDF['pickle_path']= taskDF['task'].str[17:]
    # taskDF['pickle_fn'] = taskDF['pickle_path'].apply(lambda x: os.path.basename(x))
    m = taskDF.merge(outDF, left_on='ID_task', right_on='ID_model', how='outer', suffixes=['', '_'], indicator=True)
    # The indicator=True setting is useful as it adds a column called _merge, with all changes between df1 and df2, categorized into 3 possible kinds: "left_only", "right_only" or "both".
    missing = m[m['_merge'] == 'left_only']
    #save task_arguments1.txt with missing file only
    fn = os.path.join(os.path.dirname(full_path_to_task_argument),'task_arguments1.txt')
    missing[['ID_task', 'pickle_path']].to_csv(fn, index=False, sep=' ', header=None)
    present = m[m['_merge'] == 'both']
    print('Total number of missing output =' + str(missing['ID_task'].count()))
    print('List of missing in output:')
    print(missing['ID_task'])
    #print('Missing model spec')
    print('Total number of present output =' + str(present['ID_task'].count()))
    print('List of present:')
    print(present['ID_task'])
    # pkl_list = missing['pickle_fn'].tolist()
    # for fn in pkl_list:
    #     pic = pd.read_pickle(pkl_path+'/'+fn)
    #     print(pic)

    print('End')

if __name__ == '__main__':
    check_outputs(r'X:\PY_data\ML1_data_output\ZAsummer\Model\condor_AUG2022\20220715\20220715\output',r'X:\PY_data\ML1_data_output\ZAsummer\Model\condor_AUG2022\20220715\20220715\task_arguments_all.txt')
    # check_outputs('/eos/jeodpp/data/projects/ML4CAST/condor/ML1_data_output/ZAsummer/Model/20220715/output/', '/eos/jeodpp/data/projects/ML4CAST/condor/ML1_data_output/ZAsummer/task_arguments.txt')