import pandas as pd
import matplotlib.pyplot as plt
import os
import src.constants as cst
import glob
import numpy as np
import b05_Init
import Model_error_stats

# Now for the best model and benchmark make scatter plot
# teh refrence data is b1

def plot_best_scatter(dir, target):
    project = b05_Init.init(target)
    df_regNames = pd.read_csv(project['input_dir'] + '/CountryStats' + '/' + project['AOI'] + '_REGION_id.csv')


    b1 =  pd.read_csv(dir + '/' + 'all_model_best1.csv')

    base_dir = os.path.join(dir, 'scatter_best_models')
    try:
        os.makedirs(base_dir)
    except FileExistsError:
        pass

    crops = b1['Crop'].unique()
    forcTimes = b1['forecast_time'].unique()
    for c in crops:
        c_dir =  os.path.join(base_dir, c)
        try:
            os.makedirs(c_dir)
        except FileExistsError:
            pass
        for t in forcTimes:
            df_c_t = b1[(b1['Crop']==c) & (b1['forecast_time']==t)]
            # get the run_ids, put  first
            Ml_run_id = df_c_t[~df_c_t['Estimator'].isin(cst.benchmarks)]['runID']
            mres_fns = glob.glob(os.path.join(dir, '*'+Ml_run_id.values[0]+'*mres.csv'))

            bench_run_id = df_c_t[df_c_t['Estimator'].isin(cst.benchmarks)]['runID']
            model_labels = ['ML'] + df_c_t[df_c_t['Estimator'].isin(cst.benchmarks)]['Estimator'].tolist()

            for b in bench_run_id.values:
                tmp = glob.glob(os.path.join(dir, '*'+b+'*mres.csv'))
                mres_fns = mres_fns + tmp

            fig, axs = plt.subplots(2,2, figsize=(10, 10), constrained_layout=True)

            for i, ax in enumerate(axs.flatten()):
                #open mres
                df = pd.read_csv(mres_fns[i])
                lims = [np.floor(np.min([df['yLoo_true'].values, df['yLoo_pred'].values])),
                        np.ceil(np.max([df['yLoo_true'].values, df['yLoo_pred'].values]))]
                r2p = Model_error_stats.r2_nan(df['yLoo_true'].values, df['yLoo_pred'].values)
                for au_code in df['AU_code'].unique():
                    x = df[df['AU_code']==au_code]['yLoo_true'].values
                    y = df[df['AU_code'] == au_code]['yLoo_pred'].values
                    lbl = df_regNames[df_regNames['AU_code']==au_code.astype('int')]['AU_name'].values[0]
                    ax.scatter(x,y, label=lbl)
                    ax.plot(lims, lims, color='black', linewidth=0.5)
                    ax.set_title(model_labels[i] + ',R2p='+str(np.round(r2p,2)))
                    ax.set_xlim(lims)
                    ax.set_ylim(lims)
                    ax.set_xlabel('Obs')
                    ax.set_ylabel('Pred')
                    ax.legend(frameon=False, loc='upper left')

            fig.savefig(os.path.join(c_dir, c + 'forecst_time_'+str(t)+'.png'))
            plt.close(fig)

    print('end')