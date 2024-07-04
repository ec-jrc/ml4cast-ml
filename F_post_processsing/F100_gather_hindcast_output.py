import pathlib
import pandas as pd

def gather_output(dir):
    run_res = list(sorted(pathlib.Path(dir).glob('ID*_output.csv')))
    print('N files = ' + str(len(run_res)))
    print('Missing files are printed ') #(no warning issued if they are files that were supposed to be skipped (ft sel asked on 1 var)')


    cc = 0
    if len(run_res) > 0:
        for file_obj in run_res:
            #print(file_obj, cc)
            cc = cc + 1
            try:
                df = pd.read_csv(file_obj)
            except:
                print('Empty file ' + str(file_obj))
            else:
                try:
                    run_id = int(df['runID'][0]) #.split('_')[1])
                except:
                    print('Error in the file ' + str(file_obj))
                else:
                    # date_id = str(df['runID'][0].split('_')[0])
                    # df_updated = df

                    if file_obj == run_res[0]:
                        #it is the first, save with hdr
                        df.to_csv(file_obj.parent / 'all_model_output.csv', mode='w', header=True, index=False)
                    else:
                        #it is not first, without hdr
                        df.to_csv(file_obj.parent / 'all_model_output.csv', mode='a', header=False, index=False)
                        # print if something is missing
                        if run_id > run_id0:
                            if (run_id != run_id0 + 1):
                                for i in range(run_id0 + 1, run_id):
                                    print('Non consececutive runids:' + str(i))
                        # else:
                        #     print('Date changed?', date_id, 'new run id', run_id0)
                    run_id0 = run_id
    else:
        print('There is no a single output file')
