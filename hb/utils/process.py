import QuantLib as ql
from typing import Union
from collections import namedtuple
from hb.utils.date import *
from hb.utils.termstructure import *
import pandas as pd
import os

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

def save_process_param(dir_, param):
    """Save process parameter into directory as csv

    Args:
        dir_ (str): Directory for saving
        param (Union[GBMProcessParam, HestonProcessParam]): process parameter to be saved
    """
    if isinstance(param, HestonProcessParam):
        to_file = os.path.join(dir_, 'HestonProcessParam.csv')
    elif isinstance(param, GBMProcessParam):
        to_file = os.path.join(dir_, 'GBMProcessParam.csv')
    else:
        raise NotImplementedError()
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    pd.DataFrame({f: [param.__getattribute__(f)] for f in param._fields}) \
        .to_csv(to_file, index=False)

def load_process_param(dir_):
    """Load process parameter from csv

    Args:
        dir_ (str): directory
        
    Return:
        (Union[GBMProcessParam, HestonProcessParam]): loaded process param
    """
    if os.path.exists(os.path.join(dir_, 'HestonProcessParam.csv')):
        param_df = pd.read_csv(os.path.join(dir_, 'HestonProcessParam.csv'))
        return HestonProcessParam(
            risk_free_rate=param_df['risk_free_rate'].values[0],
            spot=param_df['spot'].values[0], 
            drift=param_df['drift'].values[0], 
            dividend=param_df['dividend'].values[0],
            spot_var=param_df['spot_var'].values[0], 
            kappa=param_df['kappa'].values[0], 
            theta=param_df['theta'].values[0], 
            rho=param_df['rho'].values[0], 
            vov=param_df['vov'].values[0], 
            use_risk_free=param_df['use_risk_free'].values[0]
        )
    elif os.path.exists(os.path.join(dir_, 'GBMProcessParam.csv')):
        param_df = pd.read_csv(os.path.join(dir_, 'GBMProcessParam.csv'))
        return GBMProcessParam(
            risk_free_rate = param_df['risk_free_rate'].values[0],
            spot=param_df['spot'].values[0], 
            drift=param_df['drift'].values[0], 
            dividend=param_df['dividend'].values[0], 
            vol=param_df['vol'].values[0], 
            use_risk_free=param_df['use_risk_free'].values[0]
        )
    else:
        return None
        
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

if __name__ == "__main__":
    heston_param = HestonProcessParam(
            risk_free_rate=0.015,
            spot=100, 
            drift=0.15, 
            dividend=0.0,
            spot_var=0.096024, kappa=6.288453, theta=0.397888, 
            rho=-0.696611, vov=0.753137, use_risk_free=False
        )
    save_process_param('./process_param', heston_param)
    param = load_process_param('./process_param')
    print(param)