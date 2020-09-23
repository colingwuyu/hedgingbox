import QuantLib as ql
from typing import Union
from collections import namedtuple
from hb.utils.date import *
from hb.utils.termstructure import *
import pandas as pd
import os
import json

class GBMProcessParam(object):
    __slots__ = ('risk_free_rate', 'spot', 'drift', 'dividend',
                 'vol', 'use_risk_free')
    _fields = ('risk_free_rate', 'spot', 'drift', 'dividend',
               'vol', 'use_risk_free')

    def __init__(self, risk_free_rate=None, spot=None, 
                 drift=None, dividend=None,
                 vol=None, use_risk_free=None):
        self.risk_free_rate = risk_free_rate
        self.spot = spot
        self.drift = drift
        self.dividend = dividend
        self.vol = vol
        self.use_risk_free = use_risk_free

    @classmethod
    def load_json(cls, json_: Union[str, dict]):
        if isinstance(json_, str):
            dict_json = json.loads(json_)
        else:
            dict_json = json_
        ret_obj = cls()
        ret_obj.spot = dict_json["spot"]
        ret_obj.drift = dict_json["drift"]
        ret_obj.dividend = dict_json["dividend"]
        ret_obj.vol = dict_json["vol"]
        return ret_obj

    def jsonify_dict(self) -> dict:
        dict_json = dict()
        dict_json["spot"] = self.spot
        dict_json["drift"] = self.drift
        dict_json["dividend"] = self.dividend
        dict_json["vol"] = self.vol
        return dict_json

    def __repr__(self):
        return json.dumps(self.jsonify(), indent=4)

class HestonProcessParam(object):
    __slots__ = ('risk_free_rate', 'spot', 'spot_var', 'drift',
                 'dividend', 'kappa', 'theta', 'rho', 'vov', 'use_risk_free')
    _fields = ('risk_free_rate', 'spot', 'spot_var', 'drift',
               'dividend', 'kappa', 'theta', 'rho', 'vov', 'use_risk_free')

    def __init__(self, risk_free_rate=None,
                 spot=None, spot_var=None, drift=None,
                 dividend=None, kappa=None, theta=None, 
                 rho=None, vov=None, use_risk_free=None):
        self.risk_free_rate = risk_free_rate
        self.spot = spot
        self.spot_var = spot_var
        self.drift = drift
        self.dividend = dividend
        self.kappa = kappa
        self.theta = theta
        self.rho = rho
        self.vov = vov
        self.use_risk_free = risk_free_rate

    @classmethod
    def load_json(cls, json_: Union[str, dict]):
        if isinstance(json_, str):
            dict_json = json.loads(json_)
        else:
            dict_json = json_
        ret_obj = cls()
        ret_obj.spot = dict_json["spot"]
        ret_obj.spot_var = dict_json["spot_var"]
        ret_obj.drift = dict_json["drift"]
        ret_obj.dividend = dict_json["dividend"]
        ret_obj.kappa = dict_json["kappa"]
        ret_obj.theta = dict_json["theta"]
        ret_obj.rho = dict_json["rho"]
        ret_obj.vov = dict_json["vov"]
        return ret_obj

    def jsonify_dict(self) -> dict:
        dict_json = dict()
        dict_json["spot"] = self.spot
        dict_json["spot_var"] = self.spot_var
        dict_json["drift"] = self.drift
        dict_json["dividend"] = self.dividend
        dict_json["kappa"] = self.kappa
        dict_json["theta"] = self.theta
        dict_json["rho"] = self.rho
        dict_json["vov"] = self.vov
        return dict_json

    def __repr__(self):
        return json.dumps(self.jsonify(), indent=4)

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