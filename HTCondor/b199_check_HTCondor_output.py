import glob
import pandas as pd
import os

def check_outputs (dir_output, full_path_to_task_argument, pkl_path):
    tmp = glob.glob(dir_output + '/*_output.csv')
    outputList = [os.path.basename(x) for x in tmp]
    outDF = pd.DataFrame(outputList, columns=['fn'])
    outDF['ID_model'] = outDF['fn'].str[3:18]

    with open(full_path_to_task_argument) as f:
        lines = [line.rstrip() for line in f]
    taskDF = pd.DataFrame(lines, columns=['task'])
    taskDF['ID_task'] = taskDF['task'].str[0:15]
    taskDF['pickle_path']= taskDF['task'].str[17:]
    taskDF['pickle_fn'] = taskDF['pickle_path'].apply(lambda x: os.path.basename(x))



    m = taskDF.merge(outDF, left_on='ID_task', right_on='ID_model', how='outer', suffixes=['', '_'], indicator=True)
    # The indicator=True setting is useful as it adds a column called _merge, with all changes between df1 and df2, categorized into 3 possible kinds: "left_only", "right_only" or "both".
    missing = m[m['_merge'] == 'left_only']
    print('Total numeber of missing output =' + str(missing['task'].count()))
    print('List of task not present in output')
    print(missing)
    print('Missing model spec')
    pkl_list = missing['pickle_fn'].tolist()
    for fn in pkl_list:
        pic = pd.read_pickle(pkl_path+'/'+fn)
        print(pic)

    print('End')
