import QuantLib as ql
import numpy as np

day_count = ql.Actual360()
DAYS_PER_YEAR = 360.
calendar = ql.NullCalendar()
ql.Settings.instance().evaluationDate = ql.Date(1,1,2000)

IMPLIED_VOL_FLOOR = 0.0001
np_dtype = np.float32
MIN_FLT_VALUE = np.finfo('float32').min
MAX_FLT_VALUE = np.finfo('float32').max
LARGE_NEG_VALUE = -1e-12