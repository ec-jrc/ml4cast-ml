import src.constants as cst
import pandas as pd
import matplotlib.pyplot as plt
#from vam.whittaker import ws2d
from preprocess.whittaker import whittaker_smooth_m_withWeights, whittaker_smooth_m
import os
import numpy as np

# Smooth the SM data with Whittaker using percent cover as weight (so more% cover gets more credit)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


input_file = cst.sm_file_for_b10 #your full path to input here
df = pd.read_csv(input_file)
df = df[(df['classset_name'] == 'static masks') & (df['class_name'] == 'crop')]
# add a column for the smoothed sm
df['sm_smooth'] = -9999
# make two df, one with all except sm (untouched) and one with sm only
df_other = df[df['variable_name'] != 'soil_moisture']
df_sm = df[df['variable_name'] == 'soil_moisture']
df_sm_out = df_sm[0:0]
#work on sm
regions = df_sm['reg0_name'].unique()
for reg in regions:
    print(reg)
    df_reg = df_sm[df_sm['reg0_name']==reg]
    y = np.array(df_reg['mean'].values).astype('double')
    w  = np.array(df_reg['afi_pct'].values/100).astype('double')
    # z = ws2d(y, 0.5, w)
    #w = w *0 +1
    #z = whittaker_smooth_m(y, 0.5, None, d=2)
    z = whittaker_smooth_m_withWeights(y, 0.5, w, None, d=2)
    df_reg['sm_smooth'] = z
    df_sm_out = pd.concat([df_sm_out, df_reg])
    # Test to understand lambda, lambda fixed at 0.5
    plt.figure(figsize=(30,10))
    a = 0
    b = len(y)
    xax1 = range(a,b)
    ax1 = plt.subplot()
    l1, = ax1.plot(xax1, w[a:b], color='black')
    ax1.set_ylim([0,1])
    ax1.set_xlim([0, 50])
    ax2 = ax1.twinx()
    l2, = ax2.plot(xax1, y[a:b], color='grey')
    l2, = ax2.plot(xax1, z[a:b], color='blue')
    ax2.set_ylim([0, 0.4])
    ax2.set_xlim([0, 50])
    # l2, = ax2.plot(xax1, zz[a:b], color='red')
    fn = os.path.join(os.path.dirname(cst.sm_file_for_b10), reg + '_smoothed.png')
    plt.tight_layout
    plt.savefig(fn, bbox_inches='tight')

#replace original mean with smoothed
df_sm_out['mean'] = df_sm_out['sm_smooth']
#reconcile in one single df
df_sm_out = pd.concat([df_sm_out, df_other])
#drop smoothed
df_sm_out = df_sm_out.drop(['sm_smooth'], axis=1)

basename_without_ext = os.path.splitext(os.path.basename(cst.sm_file_for_b10))[0]
fn = os.path.join(os.path.dirname(cst.sm_file_for_b10), basename_without_ext+'_smoothed.csv')
df_sm_out.to_csv(fn, index=False)

