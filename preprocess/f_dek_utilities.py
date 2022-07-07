def f_datetime2dek(x):
    #return the dek into with the date x falls
    day = x.day
    month = x.month
    if month > 12:
        return -1
    if ((day >= 1) and (day <= 10)):
        dek = 1 + (month-1) * 3
    elif ((day >= 11) and (day <= 20)):
        dek = 2 + (month - 1) * 3
    elif ((day >= 21) and (day <= 31)):
        dek = 3 + (month - 1) * 3
    else:
        dek = -1
    return dek


from datetime import datetime

def f_dek_year2dateFirstDekDay(d,y):
    dates = [datetime(y, 1, 1), datetime(y, 1, 11), datetime(y, 1, 21),
             datetime(y, 2, 1), datetime(y, 2, 11), datetime(y, 2, 21),
             datetime(y, 3, 1), datetime(y, 3, 11), datetime(y, 3, 21),
             datetime(y, 4, 1), datetime(y, 4, 11), datetime(y, 4, 21),
             datetime(y, 5, 1), datetime(y, 5, 11), datetime(y, 5, 21),
             datetime(y, 6, 1), datetime(y, 6, 11), datetime(y, 6, 21),
             datetime(y, 7, 1), datetime(y, 7, 11), datetime(y, 7, 21),
             datetime(y, 8, 1), datetime(y, 8, 11), datetime(y, 8, 21),
             datetime(y, 9, 1), datetime(y, 9, 11), datetime(y, 9, 21),
             datetime(y, 10, 1), datetime(y, 10, 11), datetime(y, 10, 21),
             datetime(y, 11, 1), datetime(y, 11, 11), datetime(y, 11, 21),
             datetime(y, 12, 1), datetime(y, 12, 11), datetime(y, 12, 21)]
    return dates[d-1]