import datetime as dt
import pandas as pd
import numpy as np
import itertools
import os
from dynamic_masking import gaussian_mixture

CSV_FN = 'to be def'
DIR_OUT = 'to be def'
NBINS = 100
NMAXCOMPONENTS = 2
AU_NAME = 'Khersons‘ka'
DATE='2019-03-11'
CROP_TYPE_NAME = 'Winter wheat'
AIC_THRESHOLD = -200

pd.set_option('display.max_columns', None)
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)

def main():
    df = pd.read_csv(CSV_FN)
    # x is 2 columns NDVI values and number of pixels with that value
    # modify so that data contain NDVI values

    data = [list(itertools.repeat(x, y)) for x, y in zip(df['ndvi_val'], df['pixel_count'])]
    data = np.array(list(itertools.chain(*data)))
    binValues = np.linspace(0, 1, NBINS, endpoint=True)
    hist, bin = np.histogram(data, bins=binValues)
    fig, responsibilities = gaussian_mixture.gam(hist, NMAXCOMPONENTS, AU_NAME, DATE, bins=bin,
                                                 aic_threshold=AIC_THRESHOLD)
    fn = os.path.join(DIR_OUT,
                      AU_NAME + '_' + CROP_TYPE_NAME + '_NDVI_zgam' + str(NMAXCOMPONENTS) + '_' + str(DATE) + '.png')
    fig.savefig(fn)

    print('end')


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    desired_width = 700
    pd.set_option('display.width', desired_width)
    start = dt.datetime.now()
    main()
    print('Execution time:', dt.datetime.now() - start)