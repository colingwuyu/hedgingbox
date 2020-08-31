import QuantLib as ql
from math import pow, sqrt
import numpy as np
from scipy.optimize import root
from scipy.optimize import basinhopping
from hb.utils.date import *
from hb.utils.process import HestonProcessParam


class MyBounds(object):
     def __init__(self, xmin=[0.,0.01,0.01,-1,0], xmax=[1,15,1,1,1.0] ):
         self.xmax = np.array(xmax)
         self.xmin = np.array(xmin)
     def __call__(self, **kwargs):
         x = kwargs["x_new"]
         tmax = bool(np.all(x <= self.xmax))
         tmin = bool(np.all(x >= self.xmin))
         return tmax and tmin

def cost_function_generator(model, helpers,norm=False):
    def cost_function(params):
        params_ = ql.Array(list(params))
        model.setParams(params_)
        error = [h.calibrationError() for h in helpers]
        if norm:
            return np.sqrt(np.sum(np.abs(error)))
        else:
            return error
    return cost_function

def calibration_report(helpers, grid_data, detailed=False):
    avg = 0.0
    if detailed:
        print("%15s %25s %15s %15s %20s" % (
            "Strikes", "Expiry", "Market Value", 
             "Model Value", "Relative Error (%)"))
        print("="*100)
    for i, opt in enumerate(helpers):
        err = (opt.modelValue()/opt.marketValue() - 1.0)
        date,strike = grid_data[i]
        if detailed:
            print("%15.2f %25s %14.5f %15.5f %20.7f " % (
                strike, str(date), opt.marketValue(), 
                opt.modelValue(), 
                100.0*(opt.modelValue()/opt.marketValue() - 1.0)))
        avg += abs(err)
    avg = avg*100.0/len(helpers)
    if detailed: print("-"*100)
    summary = "Average Abs Error (%%) : %5.9f" % (avg)
    print(summary)
    return avg
    
def heston_calibration(risk_free_rate, stock, 
                       calls, init_condition=(0.02,0.2,0.5,0.1,0.01)):
    calculation_date = date0
    _yield_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, risk_free_rate, day_count))
    _dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, stock.get_dividend_yield(), day_count))
    _spot = stock.get_quote()
    expiration_dates = []
    strikes = []
    data = []
    add_strikes = True
    for c_exp in calls:
        expiration_dates = expiration_dates + [c_exp[0].get_maturity_date()]
        for c in c_exp:
            if add_strikes:
                strikes = strikes + [c.get_strike()]
        data.append([c.get_quote() for c in c_exp])
        add_strikes = False
    theta, kappa, sigma, rho, v0 = init_condition
    process = ql.HestonProcess(_yield_ts, _dividend_ts, 
                           ql.QuoteHandle(ql.SimpleQuote(_spot)), 
                           v0, kappa, theta, sigma, rho)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model) 
    heston_helpers = []
    grid_data = []
    for i, date in enumerate(expiration_dates):
        for j, s in enumerate(strikes):
            t = (date - calculation_date )
            p = ql.Period(t, ql.Days)
            vols = data[i][j]
            helper = ql.HestonModelHelper(
                p, calendar, _spot, s, 
                ql.QuoteHandle(ql.SimpleQuote(vols)),
                _yield_ts, _dividend_ts)
            helper.setPricingEngine(engine)
            heston_helpers.append(helper)
            grid_data.append((date, s))
    initial_condition = list(model.params())
    bounds = [(0,1),(0.01,15), (0.01,1.), (-0.8,0.8), (0,1.0) ]

    mybound = MyBounds()
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
    cost_function = cost_function_generator(
        model, heston_helpers, norm=True)
    sol = basinhopping(cost_function, initial_condition, niter=5,
                    minimizer_kwargs=minimizer_kwargs,
                    stepsize=0.005,
                    accept_test=mybound,
                    interval=10)
    theta, kappa, sigma, rho, v0 = model.params()
    print(stock.get_name(), "Calibrated Heston Parameters: theta = %f, kappa = %f, vov = %f, rho = %f, spot_var = %f" % \
        (theta, kappa, sigma, rho, v0))
    error = calibration_report(heston_helpers, grid_data)
    return HestonProcessParam(
                spot=_spot, 
                drift=stock.get_annual_yield(), 
                dividend=stock.get_dividend_yield(),
                spot_var=v0, kappa=kappa, theta=theta, rho=rho, vov=sigma
            )

if __name__ == "__main__":
    from hb.transaction_cost.percentage_transaction_cost import PercentageTransactionCost
    from hb.utils.date import *
    from hb.utils.consts import *
    from hb.instrument.instrument_factory import InstrumentFactory
    risk_free_rate = 0.015
    spx = InstrumentFactory.create(
        'Stock AMZN 3400 25 0 0.15'
    )
    maturity = ['1W', '2W', '3W', '4W', '7W', '3M']
    strike = [3395, 3400, 3405, 3410]
    iv = [[33.19, 33.21, 33.08, 33.34],
          [36.08, 36.02, 36.11, 35.76],
          [38.14, 38.08, 37.89, 37.99],
          [39.99, 39.84, 40.01, 39.79],
          [43.51, 43.7, 43.67, 43.63],
          [49.34, 49.25, 48.77, 48.56]]
    spx_listed_calls = []
    n = 0
    for i, m in enumerate(maturity):
        spx_listed_calls_m = []
        for j, s in enumerate(strike):
            spx_listed_calls_m = spx_listed_calls_m \
                    + [InstrumentFactory.create(
                        f'EuroOpt AMZN Listed {m} Call {s} {iv[i][j]} 5 (AMZN_Call{n})'
                    ).underlying(spx)] 
            n += 1
        spx_listed_calls.append(spx_listed_calls_m)
    heston_calibration(risk_free_rate, spx, spx_listed_calls)
