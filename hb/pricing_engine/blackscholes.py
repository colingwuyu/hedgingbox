import math
from scipy.stats import norm


def price(call, s0, r, q, strike, sigma, tau_e, tau_d):
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


def delta(call, s0, r, q, strike, sigma, tau_e, tau_d):
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
