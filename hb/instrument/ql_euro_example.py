import QuantLib as ql # version 1.5
import matplotlib.pyplot as plt

# option data
maturity_date = ql.Date(15, 1, 2016)
spot_price = 127.62
strike_price = 130
sigma = 0.30 # the historical vols for a year
dividend_rate =  0.0163
option_type = ql.Option.Call

risk_free_rate = 0.001
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates()

calculation_date = ql.Date(8, 5, 2015)
ql.Settings.instance().evaluationDate = calculation_date

# construct the European Option
payoff = ql.PlainVanillaPayoff(option_type, strike_price)
exercise = ql.EuropeanExercise(maturity_date)
european_option = ql.VanillaOption(payoff, exercise)

spot_handle = ql.QuoteHandle(
    ql.SimpleQuote(spot_price)
)
flat_ts = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, risk_free_rate, day_count)
)
dividend_yield = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, dividend_rate, day_count)
)
flat_vol_ts = ql.BlackVolTermStructureHandle(
    ql.BlackConstantVol(calculation_date, calendar, sigma, day_count)
)
bsm_process = ql.BlackScholesMertonProcess(spot_handle, 
                                           dividend_yield, 
                                           flat_ts, 
                                           flat_vol_ts)

european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
bs_price = european_option.NPV()
print("The theoretical price is ", bs_price)

v0=0.09
rho=-0.4
vov=3
kappa=0.2
theta=0.09
heston_process = ql.HestonProcess(flat_ts, dividend_yield,
                                  spot_handle, v0,
                                  kappa,theta,vov,rho)

def binomial_price(bsm_process, steps):
    binomial_engine = ql.BinomialVanillaEngine(bsm_process, "crr", steps)
    european_option.setPricingEngine(binomial_engine)
    return european_option.NPV()

def heston_price(heston_process):
    engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process),0.01, 1000)
    european_option.setPricingEngine(engine)
    h_price = european_option.NPV()
    return h_price

def bs_price(bsm_process):
    bs_engine = ql.AnalyticEuropeanEngine(bsm_process)
    european_option.setPricingEngine(bs_engine)
    return european_option.NPV()
