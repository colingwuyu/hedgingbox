import QuantLib as ql

day_count = ql.Actual360()
DAYS_PER_YEAR = 360.
calendar = ql.NullCalendar()
date0 = ql.Date(1,1,2000)
ql.Settings.instance().evaluationDate = ql.Date(1,1,2000)

IMPLIED_VOL_FLOOR = 0.0001
