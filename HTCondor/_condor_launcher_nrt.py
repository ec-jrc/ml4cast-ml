import sys
import time, pickle
import glob
import os
import ast
import pandas as pd

import src.constants as cst
import ml.modeller as modeller


def launcher(pckl_fn):
    # read pickle file
    with open(pckl_fn, 'rb') as f:
        uset = pickle.load(f)

    # Instantiate class for forecasting
    forecaster = modeller.YieldForecaster(uset['runID'],
                                          uset['target'],
                                          uset['cropID'],
                                          uset['algorithm'],
                                          uset['yvar'],
                                          uset['doOHE'],
                                          uset['selected_features'],
                                          uset['forecast_time'],
                                          uset['time_sampling'])
    print(forecaster)
    input_fn = uset['input_data']
    forecast_fn = os.path.join(
        forecaster.output_dir,
        f'{forecaster.id}_{forecaster.leadtime}_{forecaster.crop}_forecasts.csv'
    )

    if not os.path.exists(forecast_fn):
        X, y, groups = forecaster.preprocess()
        tic = time.time()
        forecaster.fit(X, y, groups)
        runTimeH = (time.time() - tic) / (60 * 60)
        print(f'Model fitted in {runTimeH} hours')
        X_forecast, regions = forecaster.preprocess_currentyear(input_fn)
        y_forecast = forecaster.predict(X_forecast)
        y_uncertainty = forecaster.predict_uncertainty(X, y, groups, X_forecast)
        df_forecast = pd.DataFrame({'regions': regions,
                                    'forecast': y_forecast,
                                    'uncertainty': ((y_uncertainty / y_forecast) * 100)})
        df_forecast.to_csv(forecast_fn)
    else:
        print('Output files already exist')


if __name__ == '__main__':
    print(sys.version)
    uset_file = r'{}'.format(sys.argv[1])
    launcher(pckl_fn=uset_file)
