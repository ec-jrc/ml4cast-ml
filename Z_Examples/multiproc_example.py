import datetime as dt
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

from asap_toolbox.indicators import ndvi
from asap_toolbox.util.date import string2raw, raw2dekade, dekade2string


START_DEKAD = dekade2string(ndvi.NDVI_FIRST_DEKADE)
END_DEKAD = 20220721

PROC_NUM = 36
USE_THREADING = False


def main():
    dkd_list = list(map(raw2dekade, range(string2raw(START_DEKAD), string2raw(END_DEKAD) + 1)))
    if PROC_NUM > 1:
        if USE_THREADING:
            p = ThreadPool(PROC_NUM)
        else:
            p = Pool(PROC_NUM)
        p.map(process_dekade, dkd_list)
    else:
        for dkd in dkd_list:
            print(dkd)
            process_dekade(dkd, overwrite=False)


def process_dekade(dkd, overwrite=False, delete=False):
    # do something with your code
    # if needed return some output
    output = 1
    return output


if __name__ == '__main__':
    start = dt.datetime.now()
    main()
    print('Execution time:', dt.datetime.now() - start)
