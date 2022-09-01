import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import datetime

import src.constants as cst
import Model_error_stats

def CEC_model(target):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 500)
    doPlot = False
    aoi = target
    fn = cst.cec_estimates_fn
    df = pd.read_csv(fn)
    df.rename(columns = {'Yield (Tons ha-1)':'Yield_pred_CEC'}, inplace = True)
    df['DateTimeIssue'] = pd.to_datetime(df['Time of Issue'], format='%d-%m-%Y')
    # remove final; estimate (which is our ground truth). The final estimate is the last of each year
    for y in df['Year'].unique():
        last_n = df[df['Year']==y]['Forecast number'].max()
        df.drop(df.loc[(df['Forecast number']==last_n) & (df['Year']==y)].index, inplace=True)
    # add true yield

    stats = pd.read_pickle(os.path.join(cst.odir, aoi, f'{aoi}_stats.pkl'))
    stats.rename(columns = {'Yield':'Yield_obs'}, inplace = True)
    df = df.merge(stats, how='left', left_on=['Year','ASAP1_ID','Crop_ID'], right_on=['Year','ASAP1_ID','Crop_ID'])
    # now group forecasts by months (from 2 to 11). Do not use forecast number because 2016 started earlier than all other yeras (2016 started in Jan, all others in Feb)
    forecast_month = list(range(2,11))
    crop_ids = [1,2,3]
    crop_names = ['Maize_total','Sunflower','Soybeans']
    df_out = pd.DataFrame(columns=['CropID','CropName','CEC_date', 'rRMSEp'])
    for croid in crop_ids:
        rel_Pred_RMSE = []
        avg_date_forecast = []
        for formon in forecast_month:
            df_crop = df[df['Crop_ID']==croid]
            df_crop_mon = df_crop[df_crop['DateTimeIssue'].dt.month ==formon]
            df_crop_mon['DOY'] = df_crop_mon['DateTimeIssue'].dt.dayofyear
            avg_DOY = np.mean(df_crop_mon['DOY'].values)
            #make it date of a non leap year, e.g. 201

            avg_date_forecast.append(pd.to_datetime(avg_DOY-1, unit='D', origin=str(2000)))
            # get pred and obs
            mRes = pd.DataFrame(columns=['yLoo_pred', 'yLoo_true', 'AU_code', 'Year'])
            mRes['yLoo_pred'] = df_crop_mon['Yield_pred_CEC']
            mRes['yLoo_true'] = df_crop_mon['Yield_obs']
            mRes['AU_code'] = df_crop_mon['ASAP1_ID']
            mRes['Year'] = df_crop_mon['Year']


            error_AU_level = Model_error_stats.allStats(mRes)
            rel_Pred_RMSE.append(error_AU_level["rel_Pred_RMSE"])
            # plt.scatter(y_obs, y_pred)
            # plt.show()
        tmp_df = pd.DataFrame(columns=['CropID','CropName','CEC_date', 'rRMSEp'])
        tmp_df['CEC_date'] = avg_date_forecast
        tmp_df['rRMSEp'] = rel_Pred_RMSE
        tmp_df['CropID'] = croid
        tmp_df['CropName'] = crop_names[croid-1]
        if doPlot == True:
            fig, axs = plt.subplots(figsize=(12, 3), constrained_layout=True)
            xlabel = 'Time (Month-day)'
            xlim = [datetime.date(2020, 10, 15), datetime.date(2021, 12, 1)]
            fmt = mdates.DateFormatter('%b-%d')
            axs.set_ylim([0,30])
            axs.plot(avg_date_forecast, rel_Pred_RMSE, color='grey', linewidth=1, marker='o', label='CEC estimates')
            axs.xaxis.set_major_formatter(fmt)
            axs.set_xlim(xlim[0], xlim[1])
            lbl = r'${\rm rRMSE_p\/(\%)}$'
            axs.set_ylabel(lbl)
            axs.set_xlabel(xlabel)
            axs.set_title(crop_names[croid-1], fontsize=12)
            axs.legend(frameon=False, loc='upper left', ncol=len(axs.lines))
            plt.show()
        df_out = pd.concat([df_out,tmp_df])


    return df_out

if __name__ == '__main__':
    CEC_mode('ZAsummer')