import math
from scipy.stats import norm
from scipy.optimize import newton
from hb.utils.process import *
from hb.instrument.european_option import EuropeanOption
from hb.utils import consts
import numpy as np

def price(call, s0, r, q, sigma, strike, tau_e, tau_d):
    m = 1. if call else -1.
    df = math.exp(-r * tau_d)
    if tau_e < 1e-6:
        return df*max(m*(s0-strike), 0.)
    tau_sqrt = math.sqrt(tau_e)
    sigma_tau_sqrt = sigma * tau_sqrt
    fwd = s0*math.exp((r-q)*tau_e)
    d1tmp = (math.log(fwd / strike) +
             (0.5 * sigma ** 2 * tau_e))/sigma_tau_sqrt
    d2 = (d1tmp - sigma_tau_sqrt) * m
    d1 = d1tmp * m
    cnd1 = norm.cdf(d1)
    cnd2 = norm.cdf(d2)
    return df * ((fwd*cnd1*m) - (strike*cnd2*m))


def implied_vol(call, strike, tau_e, option_price, gbm_param):
    euro_opt = EuropeanOption("impl_vol", 'Call' if call else 'Put', strike,
                              tau_e, False)
    process = create_process(gbm_param)
    try:
        return euro_opt._option.impliedVolatility(option_price, process)
    except:
        return consts.IMPLIED_VOL_FLOOR

def create_euro_opt(call, strike, param, tau_e, price=None):
    if isinstance(param, GBMProcessParam):
        bsm_process = create_process(param)
    elif isinstance(param, HestonProcessParam):
        assert price is not None, "HestonProcess requires option price to generate implied volatility"
        sigma = implied_vol(call, strike, tau_e, price, 
                            GBMProcessParam(
                                risk_free_rate=param.risk_free_rate, spot=param.spot,
                                drift=param.drift, dividend=param.dividend, vol=0.2,
                                use_risk_free=True
                            ))
        bsm_process = create_process(
                            GBMProcessParam(
                                risk_free_rate=param.risk_free_rate, spot=param.spot,
                                drift=param.drift, dividend=param.dividend, vol=sigma,
                                use_risk_free=True
                            ))
    euro_opt = EuropeanOption("impl_vol", 'Call' if call else 'Put', strike,
                              tau_e, False)
    euro_opt._option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
    return euro_opt

def delta(call, strike, param, tau_e, price=None):
    delta_value = create_euro_opt(call, strike, param, tau_e, price)._option.delta()
    if np.isnan(delta_value):
        delta_value = delta_bk(True, param.spot, param.risk_free_rate, param.dividend, strike, param.vol, tau_e, tau_e)
    return delta_value

def gamma(call, strike, param, tau_e, price=None):
    return create_euro_opt(call, strike, param, tau_e, price)._option.gamma()

def vega(call, strike, param, tau_e, price=None):
    return create_euro_opt(call, strike, param, tau_e, price)._option.vega()

def delta_bk(call, s0, r, q, strike, sigma, tau_e, tau_d):
    m = 1. if call else -1.
    if tau_e < 1e-6:
        return 0.
    tau_sqrt = math.sqrt(tau_e)
    sigma_tau_sqrt = sigma * tau_sqrt
    fwd = s0*math.exp((r-q)*tau_e)
    d1tmp = (math.log(fwd / strike) +
             (0.5 * sigma ** 2 * tau_e))/sigma_tau_sqrt
    d1 = d1tmp * m
    cnd1 = norm.cdf(d1)
    return m*cnd1
