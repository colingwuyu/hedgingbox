import QuantLib as ql
from collections import namedtuple
from hb.utils.date import *

GBMProcessParam = namedtuple('GBMProcessParam',
                             'spot '
                             'drift '
                             'dividend '
                             'vol ')

HestonProcessParam = namedtuple('HestonProcessParam',
                                'spot '
                                'spot_var '
                                'drift '
                                'dividend '
                                'kappa '
                                'theta '
                                'rho '
                                'vov ')

def create_gbm_process(param: GBMProcessParam):
    """Create GBM Process

    Args:
        param (GBMProcessParam): GBM Process parameters

    Returns:
        QuantLib.BlackScholesMertonProcess: GBM Process
    """
    drift_curve = ql.FlatForward(get_date(), param.drift, day_count)
    flat_ts = ql.YieldTermStructureHandle(drift_curve)
    div_curve = ql.FlatForward(get_date(), param.dividend, day_count)
    dividend_ts = ql.YieldTermStructureHandle(div_curve)
    spot_quote = ql.SimpleQuote(param.spot)
    spot_handle = ql.QuoteHandle(spot_quote)
    flat_vol_ts = ql.BlackVolTermStructureHandle(
                        ql.BlackConstantVol(get_date(), calendar, param.vol, day_count)
                    )
    return ql.BlackScholesMertonProcess(spot_handle, 
                                        dividend_ts, 
                                        flat_ts, 
                                        flat_vol_ts)

def create_heston_process(param: HestonProcessParam):
    """Create Hestion Process

    Args:
        param (HestonProcessParam): Heston Process Parameter

    Returns:
        QuantLib.HestonProcess: Heston Process
    """
    drift_curve = ql.FlatForward(get_date(), param.drift, day_count)
    flat_ts = ql.YieldTermStructureHandle(drift_curve)
    div_curve = ql.FlatForward(get_date(), param.dividend, day_count)
    dividend_ts = ql.YieldTermStructureHandle(div_curve)
    spot_quote = ql.SimpleQuote(param.spot)
    spot_handle = ql.QuoteHandle(spot_quote)
    return ql.HestonProcess(flat_ts, dividend_ts, spot_handle, 
                            param.spot_var, param.kappa, param.theta, 
                            param.vov, param.rho)
    


