import QuantLib as ql
from typing import Union
from collections import namedtuple
from hb.utils.date import *
from hb.utils.termstructure import *

GBMProcessParam = namedtuple('GBMProcessParam',
                             'risk_free_rate '
                             'spot '
                             'drift '
                             'dividend '
                             'vol '
                             'use_risk_free '
                             )

HestonProcessParam = namedtuple('HestonProcessParam',
                                'risk_free_rate '
                                'spot '
                                'spot_var '
                                'drift '
                                'dividend '
                                'kappa '
                                'theta '
                                'rho '
                                'vov '
                                'use_risk_free ')

def create_gbm_process(param: GBMProcessParam):
    """Create GBM Process

    Args:
        param (GBMProcessParam): GBM Process parameters

    Returns:
        QuantLib.BlackScholesMertonProcess: GBM Process
    """
    if param.use_risk_free:
        flat_ts = create_flat_forward_ts(param.risk_free_rate)
    else:
        flat_ts = create_flat_forward_ts(param.drift)
    dividend_ts = create_flat_forward_ts(param.dividend)
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(param.spot))
    flat_vol_ts = create_constant_black_vol_ts(param.vol)
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
    if param.use_risk_free:
        flat_ts = create_flat_forward_ts(param.risk_free_rate)
    else:
        flat_ts = create_flat_forward_ts(param.drift)
    dividend_ts = create_flat_forward_ts(param.dividend)
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(param.spot))
    return ql.HestonProcess(flat_ts, dividend_ts, spot_handle, 
                            param.spot_var, param.kappa, param.theta, 
                            param.vov, param.rho)
    
def create_process(param: Union[GBMProcessParam, HestonProcessParam]):
    if isinstance(param, GBMProcessParam):
        process = create_gbm_process(param)
    elif isinstance(param, HestonProcessParam):
        process = create_heston_process(param)
    return process
