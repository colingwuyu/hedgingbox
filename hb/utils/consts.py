import QuantLib as ql
import numpy as np

day_count = ql.Actual360()
DAYS_PER_YEAR = 360.
calendar = ql.NullCalendar()
date0 = ql.Date(1,1,2000)
ql.Settings.instance().evaluationDate = ql.Date(1,1,2000)

IMPLIED_VOL_FLOOR = 0.0001
np_dtype = np.float32