import QuantLib as ql
from hb.utils.handler import Handler
from hb.utils.consts import *


date0 = Handler(ql.Date(1,1,2000))

def reset_date():
    ql.Settings.instance().evaluationDate = get_valuation_date()

def set_valuation_date(d):
    date0.set_obj(d)
    reset_date()

def get_valuation_date():
    return date0.get_obj()

def date_from_str(d_str: str):
    """convert to ql.Date

    Args:
        d_str (str): date in format "yyyy-mm-dd"
    """
    vd = [int(i) for i in d_str.split("-")]
    return ql.Date(vd[2],vd[1],vd[0])

def time_between(d1: ql.Date, d2: ql.Date = None) -> float:
    if d2:
        return day_count.yearFraction(d2, d1)
    else:
        return day_count.yearFraction(ql.Settings.instance().getEvaluationDate(), d1)

def days_between(d1: ql.Date, d2: ql.Date = None) -> int:
    if d2:
        return d1 - d2
    else:
        return d1 - ql.Settings.instance().getEvaluationDate()

def add_days(num_days: int, d: ql.Date = None) -> ql.Date:
    if d:
        return calendar.advance(d, num_days, ql.Days)
    else:
        return calendar.advance(ql.Settings.instance().getEvaluationDate(), num_days, ql.Days)

def add_time(time: float, d: ql.Date = None) -> ql.Date:
    return add_days(int(round(time*DAYS_PER_YEAR)),d)

def days_from_time(time: float) -> int:
    return int(round(time*DAYS_PER_YEAR))

def date_from_time(time: float, ref_date: ql.Date = None) -> ql.Date:
    return add_days(days_from_time(time), ref_date)

def time_from_days(days: int) -> float:
    return days/DAYS_PER_YEAR

def get_cur_time():
    return time_between(ql.Settings.instance().getEvaluationDate(), get_valuation_date())

def get_cur_days():
    return days_between(ql.Settings.instance().getEvaluationDate(), get_valuation_date())

def get_date():
    return ql.Settings.instance().getEvaluationDate()

def move_days(days:int = 1):
    ql.Settings.instance().evaluationDate = add_days(days)

def get_period_from_str(str_period: str) -> float:
    """convert period in string to period in time

    Args:
        str_period (str): 
            1W, 2W, 3W, 4W, 5W, 6W, 7W, 2M, 3M, 4M, 5M, 6M, 7M, 8M, 9M, 1Y

    Returns:
        float: period in time
            7/360, 14/360, 21/360, 30/360, 37/360, 44/360, 51/360, 60/360, 90/360, 120/360, 150/360, 180/360, 210/360, 240/360, 1

    """
    if str_period == '1W':
        return 7./DAYS_PER_YEAR
    if str_period == '2W':
        return 14./DAYS_PER_YEAR
    if str_period == '3W':
        return 21./DAYS_PER_YEAR
    if str_period == '4W':
        return 30./DAYS_PER_YEAR
    if str_period == '5W':
        return 37./DAYS_PER_YEAR
    if str_period == '6W':
        return 44./DAYS_PER_YEAR
    if str_period == '7W':
        return 51./DAYS_PER_YEAR
    if str_period == '2M':
        return 60./DAYS_PER_YEAR
    if str_period == '3M':
        return 90./DAYS_PER_YEAR
    if str_period == '4M':
        return 120./DAYS_PER_YEAR
    if str_period == '5M':
        return 150./DAYS_PER_YEAR
    if str_period == '6M':
        return 180./360.
    if str_period == '7M':
        return 210./DAYS_PER_YEAR
    if str_period == '8M':
        return 240./DAYS_PER_YEAR
    if str_period == '9M':
        return 270./DAYS_PER_YEAR
    if str_period == '1y':
        return 1.

def get_period_str_from_time(period: float) -> str:
    """convert period in time to period in string

    Args:
        period (float): 
            7/360, 14/360, 21/360, 30/360, 37/360, 44/360, 51/360, 60/360, 90/360, 120/360, 150/360, 180/360, 210/360, 240/360, 1

    Returns:
        str: period in string
            1W, 2W, 3W, 4W, 5W, 6W, 7W, 2M, 3M, 4M, 5M, 6M, 7M, 8M, 9M, 1Y
    """
    if abs(period - 7./DAYS_PER_YEAR) < 1e-5:
        return '1W'
    if abs(period - 14./DAYS_PER_YEAR) < 1e-5:
        return '2W'
    if abs(period - 21./DAYS_PER_YEAR) < 1e-5:
        return '3W'
    if abs(period - 30./DAYS_PER_YEAR) < 1e-5:
        return '4W'
    if abs(period - 37./DAYS_PER_YEAR) < 1e-5:
        return '5W'
    if abs(period - 44./DAYS_PER_YEAR) < 1e-5:
        return '6W'
    if abs(period - 51./DAYS_PER_YEAR) < 1e-5:
        return '7W'
    if abs(period - 60./DAYS_PER_YEAR) < 1e-5:
        return '2M'
    if abs(period - 90./DAYS_PER_YEAR) < 1e-5:
        return '3M'
    if abs(period - 120./DAYS_PER_YEAR) < 1e-5:
        return '4M'
    if abs(period - 150./DAYS_PER_YEAR) < 1e-5:
        return '5M'
    if abs(period - 180./DAYS_PER_YEAR) < 1e-5:
        return '6M'
    if abs(period - 210./DAYS_PER_YEAR) < 1e-5:
        return '7M'
    if abs(period - 240./DAYS_PER_YEAR) < 1e-5:
        return '8M'
    if abs(period - 270./DAYS_PER_YEAR) < 1e-5:
        return '9M'
    if abs(period - 1) < 1e-5:
        return '1Y' 

if __name__ == "__main__":
    print(time_between(ql.Date(31,1,2000)))
    print(days_between(ql.Date(31,1,2000)))
    print(add_days(30))
    print(days_between(add_days(30)))
    print(add_time(1.))
    print(time_between(add_time(1.)))
    for i in range(10):
        move_days()
        print(get_cur_days())
        print(get_cur_time())
    reset_date()
    print(get_cur_days())

