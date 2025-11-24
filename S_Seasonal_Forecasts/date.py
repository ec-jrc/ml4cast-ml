import math
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta


def get_cur_dekade(raw=False):
    """
    Returns the curent dekade based on the current date.
    :param raw: Flags if the dekade result will be returned in the YYYYKK format
                or in raw KKKK fromat (total number of dekades starting from 1900-01-01).
    :return: dekade in YYYYKK format or in KKKK fromat
    """
    if raw:
        return date2raw(dt.datetime.now())
    else:
        return date2dekade(dt.datetime.now())


def string2raw(string, format='%Y%m%d'):
    """
    Converts date in format specified in format and converts it to raw KKKK using intermediate dekade YYYYKK format
    :param string: string date
    :param format: python date format
    :return: Dekade in KKKK fromat, total number of dekades starting
             from 1900-01-01
    """
    l_dekade = string2dekade(string, format)
    return dekade2raw(l_dekade)


def pheno2dekade(pheno, year=dt.datetime.now().year, raw=True):
    """
    Converts pheno data, ranging from previous to next year, to dekade.
    Pheno data can be in "raw" format (1, 36, 72, 108) or non "raw" (natural) format  (-35, 1, 36, 72).
    :param pheno: Phenology value (start or end of season), in "raw" or "natural" format
        The input can be an integer or numpy array.
    :param year: The year that is considered as the current year to be able to determine previous and next years.
    :param raw: Flag to show if the pheno data is in "raw" or "natural" format.
    :return: Dekad in YYYYKK format
    """
    _pheno = np.copy(pheno).astype('int32')
    # check and convert from raw values (1-108 -> -35-72)
    if raw:
        if not np.all(((1 <= pheno) & (pheno <= 108))):
            raise ValueError(f'Pheno value {pheno} outside of bounds [1,108]')
        _pheno -= 36
    else:
        pass
        if not np.all(((-35 <= pheno) & (pheno <= 72))):
            raise ValueError(f'Pheno value {pheno} outside of bounds [-35,72]')
    # calc the dekade
    raw = dekade2raw(year * 100) + _pheno
    dekade = raw2dekade(raw)
    return dekade


def dekade2pheno(dekade, year=None, raw=True):
    """
    Converts dekade data YYYYKK to phenology data based on year considered as the current year.
    With the current year it is determined if the input dekade is in the previous, current or the next year.
    :param dekade: Dekade in YYYYKK format.
        The input can be an integer or numpy array.
    :param year: The year that is considered as the current year to be able to determine previous and next years.
    :param raw: Flag to show if the pheno data will be in "raw" or "natural" format.
    :return: Phenology value (start or end of season), in "raw" or "natural" format
    """
    l_year = np.floor(dekade / 100).astype(np.uint32)
    l_yearly_dekade = dekade - l_year * 100
    _year = year if year else l_year
    pheno = l_yearly_dekade - (_year - l_year) * 36
    if raw:
        pheno += 36
    return pheno


def string2dekade(string, format='%Y%m%d'):
    """
    Converts date in format specified in format and converts it to dekade YYYYKK
    :param string: string date
    :param format: python date format
    :return: Dekade in YYYYKK format
    """
    l_date = dt.datetime.strptime(str(string), format)
    return date2dekade(l_date)


def dekade2string(dekade, format='%Y%m%d'):
    """
    Converts a dekade to date at the beginning of the dekade
    and outputs a string representation of that date if specified format.
    :param dekade: Dekade in YYYYKK format
    :param format: string python datetime format string
    :return: string
    """
    l_date = dekade2date(dekade)
    return l_date.strftime(format)


def date2dekade(date):
    """
    Converts date to dekade YYYYKK
    :param date: datetiem.date date
    :return: Dekade in YYYYKK format
    """
    year_part = date.year * 100
    month_part = (date.month - 1) * 3
    day_part = math.floor((date.day - 1) / 10 + 1)
    day_part = min(day_part, 3)  # limit the day part to 3 for the 31st in the month
    return year_part + month_part + day_part


def dekade2date(dekade):
    """
    Calculates the date of the beginning of the given dekade
    :param dekade: dekade in YYYYKK format
    :return: datetime.date
    """
    l_year = math.floor(dekade / 100)
    l_yearly_dekade = dekade - l_year * 100
    l_month = math.floor((l_yearly_dekade - 1) / 3) + 1
    l_monthly_dekade = l_yearly_dekade - (l_month - 1) * 3
    l_day = (l_monthly_dekade - 1) * 10 + 1
    return dt.date(l_year, l_month, l_day)


def dekade2date_arr(dekade):
    """
    Calculates the date of the beginning of the given dekade on the input numpy array
    :param dekade: ndarray with dekade in YYYYKK format
    :return: ndarray np.datetime64
    """
    l_year = np.floor(dekade / 100)
    l_yearly_dekade = dekade - l_year * 100
    l_month = np.floor((l_yearly_dekade - 1) / 3) + 1
    l_monthly_dekade = l_yearly_dekade - (l_month - 1) * 3
    l_day = (l_monthly_dekade - 1) * 10 + 1
    # convert to datetime
    l_out = (l_year - 1970).astype('datetime64[Y]') + (l_month - 1).astype('timedelta64[M]') + (l_day - 1).astype('timedelta64[D]')
    return l_out


def raw2dekade(raw):
    """
    Converts raw KKKK to dekade YYYYKK
    :param raw: Dekade in raw KKKK fromat, total number of dekades starting from 1900-01-01.
        The input can be an integer or numpy array.
    :return: Dekade in YYYYKK format
    """
    _year = np.floor((raw - 1) / 36).astype(np.uint32)
    l_year = _year + 1900
    l_yearly_dekade = raw - _year * 36
    return l_year * 100 + l_yearly_dekade


def dekade2raw(dekade):
    """
    Converts dekade YYYYKK to raw KKKK
    :param dekade: Dekade in YYYYKK format.
        The input can be an integer or numpy array.
    :return: Dekade in KKKK fromat, total number of dekades starting from 1900-01-01
    """
    l_year = np.floor(dekade / 100).astype(np.uint32)
    l_yearly_dekade = dekade - l_year * 100
    return (l_year - 1900) * 36 + l_yearly_dekade


def date2raw(date):
    """
    Converts date to raw KKKK
    :param date: datetiem.date date
    :return: Dekade in KKKK fromat, total number of dekades starting from 1900-01-01
    """
    year_part = (date.year - 1900) * 36
    month_part = (date.month - 1) * 3
    day_part = math.floor((date.day - 1) / 10 + 1)
    day_part = min(day_part, 3)  # limit the day part to 3 for the 31st in the month
    return year_part + month_part + day_part


def raw2date(raw):
    """
    Calculates the date of the beginning of the given dekade in raw KKKK format
    :param raw: Dekade in raw KKKK fromat, total number of dekades starting from 1900-01-01
    :return: datetime.date
    """
    _year = math.floor((raw - 1) / 36)
    l_year = _year + 1900
    l_yearly_dekade = raw - _year * 36
    l_month = math.floor((l_yearly_dekade - 1) / 3) + 1
    l_monthly_dekade = l_yearly_dekade - (l_month - 1) * 3
    l_day = (l_monthly_dekade - 1) * 10 + 1
    return dt.date(l_year, l_month, l_day)


def get_dekade_last_day(date):
    """
    Rertuns the last day of the dekade
    :param date: datetime.datetime Object
    :return: datetime.datetime Object
    """
    return add_relativedelta(date, dekads=1, days=-1)


def get_dekade_days(date):
    """
    Rertuns the duration in days for the dekade
    :param date: datetime.datetime Object
    :return: datetime.datetime Object
    """
    dekad_start_date = raw2date(date2raw(date))
    next_dekad_start_date = add_relativedelta(date, dekads=1)
    return (next_dekad_start_date - dekad_start_date).days


def add_relativedelta(date, **kwargs):
    """
    Expands the dateutil.relativedelta.relativedelta function to use 'dekads' and 'dekad' parameters.
    First handles set exact then add/sustract.
    Use of dekad or dekads flattens the date to the begging of the dekad.
    Use of 'dekad' param overides the 'month' param
    :param date: datetime.datetime Object
    :param kwargs:
        year, month, day, hour, minute, second, microsecond:
            Absolute information (argument is singular); adding or subtracting a
            relativedelta with absolute information does not perform an arithmetic
            operation, but rather REPLACES the corresponding value in the
            original datetime with the value(s) in relativedelta.

        years, months, weeks, days, hours, minutes, seconds, microseconds:
            Relative information, may be negative (argument is plural); adding
            or subtracting a relativedelta with relative information performs
            the corresponding arithmetic operation on the original datetime value
            with the information in the relativedelta.
    :return: datetime.datetime Object
    """
    if not any(x in kwargs for x in ['dekad', 'dekads']):
        return date + relativedelta(**kwargs)
    else:
        # first handle years and months
        year_month_keys = ['years', 'year', 'months', 'month']
        year_month_kwargs = {key: value for (key, value) in kwargs.items() if key in year_month_keys}
        _date = date + relativedelta(**year_month_kwargs)
        _dekad = date2dekade(_date)
        # set dekad
        if 'dekad' in kwargs:
            _dekad = math.floor(_dekad / 100) * 100 + kwargs['dekad']
        # add/substract dekad
        if 'dekads' in kwargs:
            _dekad_raw = dekade2raw(_dekad) + kwargs['dekads']
            _dekad = raw2dekade(_dekad_raw)
        # convert back to date and apply rest of the params
        _date = dekade2date(_dekad)
        applied_keys = year_month_keys + ['dekads', 'dekad']
        non_applied_kwargs = {key: value for (key, value) in kwargs.items() if key not in applied_keys}
        return _date + relativedelta(**non_applied_kwargs)


def kk2mmdd(kk):
    """
    converts dekad number (1 to 36) into MMDD format
    :param kk: dekad to be converted in KK format
    :return: string for entered dekad in MMDD format
    """
    month = math.floor((kk - 1) / 3) + 1
    monthly_dekade = kk - (month - 1) * 3
    day = (monthly_dekade - 1) * 10 + 1
    return '{mm:02}{dd:02}'.format(mm=month, dd=day)


def mmdd2kk(dekad):
    """
    Converts dekad in date part string (mmdd) into KK format (1 to 36)
    :param dekad: dekad to be converted in format MMDD
    :return: int for dekad in KK format
    """
    month_part = (int(dekad[:2]) - 1) * 3
    day_part = math.floor((int(dekad[-2:]) - 1) / 10 + 1)
    return month_part + day_part