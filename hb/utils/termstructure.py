import QuantLib as ql
from typing import List
from hb.utils.date import *

def create_flat_forward_ts(r: float) -> ql.YieldTermStructureHandle:
    drift_curve = ql.FlatForward(get_date(), r, day_count)
    return ql.YieldTermStructureHandle(drift_curve)
    
def create_constant_black_vol_ts(vol: float) -> ql.BlackVolTermStructureHandle:
    return ql.BlackVolTermStructureHandle(
                        ql.BlackConstantVol(get_date(), calendar, vol, day_count)
                    )

def create_zero_curve_ts(times: List[float], rates: List[float]) -> ql.YieldTermStructureHandle:
    dates = [date_from_time(t) for t in times]
    return ql.YieldTermStructureHandle(ql.ZeroCurve(
        dates, rates, day_count, calendar
    ))


if __name__ == "__main__":
    import math
    zero_curve = create_zero_curve_ts([1., 2., 3.], [0.01, 0.02, 0.03])
    print(zero_curve.zeroRate(0.2, ql.Continuous))
    fwd_ts = create_flat_forward_ts(0.015)
    print(fwd_ts.discount(0.275), math.exp(-0.015*0.275))
    print(fwd_ts.discount(ql.Date(1,3,2000)), math.exp(-0.015*time_between(ql.Date(1,3,2000))))