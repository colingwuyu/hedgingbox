import QuantLib as ql
from hb.instrument.instrument import Instrument
from hb.transaction_cost.transaction_cost import TransactionCost
from hb.utils.date import *

class EuropeanOption(Instrument):
    def __init__(self, name: str, option_type: str, strike: float, 
                 maturity: float, tradable: bool, quote: float = None,
                 transaction_cost: TransactionCost = None,
                 underlying: Instrument = None):
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if option_type=="Call" else ql.Option.Put, 
                                       strike)
        exercise = ql.EuropeanExercise(add_time(maturity))
        self._option = ql.VanillaOption(payoff, exercise)
        self._maturity_time = maturity
        self._call = option_type=="Call"
        self._strike = strike
        self._back_up_pricing_engine = None
        super().__init__(name, tradable, quote, transaction_cost, underlying)

    def get_quote(self) -> float:
        return self._quote
    
    def set_quote(self, quote: float):
        self._quote = quote
    
    def quote(self, quote: float):
        self._quote = quote
        return self

    def get_strike(self) -> float:
        return self._strike

    def set_pricing_engine(self, pricing_engine, back_up_pricing_engine=None):
        self._option.setPricingEngine(pricing_engine)
        self._back_up_pricing_engine = back_up_pricing_engine

    def pricing_engine(self, pricing_engine, back_up_pricing_engine=None):
        self._option.setPricingEngine(pricing_engine)
        self._back_up_pricing_engine = back_up_pricing_engine
        return self

    def get_price(self):
        if abs(self._maturity_time-get_cur_time()) < 1e-5:
            # expiry
            return self.get_intrinsic_value()
        else:
            try:
                price = self._option.NPV()
                if price < self.get_intrinsic_value():
                    raise Exception("Less than intrinsic price")
                return price
            except:
                self._option.setPricingEngine(self._back_up_pricing_engine)
                return self._option.NPV()

    def get_intrinsic_value(self):
        spot, _ = self._underlying.get_price()
        if self._call:
            return 0. if spot <= self._strike else spot - self._strike
        else:
            return 0. if spot >= self._strike else self._strike - spot

    def get_maturity_time(self):
        return self._maturity_time

    def get_maturity_date(self):
        return date_from_time(self._maturity_time, ref_date=date0)

    def get_delta(self) -> float:
        return self._option.delta()

    def get_gamma(self) -> float:
        return self._option.gamma()

    def get_vega(self) -> float:
        return self._option.vega()

    def __repr__(self):
        return f'European Option {self._name}: \nunderlying=({str(self._underlying)})\noption_type={self._call}, maturity={get_period_str_from_time(self._maturity_time)}, tradable={self._tradable}, iv={self._quote}, transaction_cost={str(self._transaction_cost)}'


if __name__ == "__main__":
    from hb.transaction_cost.percentage_transaction_cost import PercentageTransactionCost
    from hb.instrument.instrument_factory import InstrumentFactory
    from hb.utils.date import *
    from hb.utils.consts import *
    from hb.utils.process import *
    spx = InstrumentFactory.create(
        'Stock AMZN 3400 25 0 0.15'
    )
    print(spx)
    spx_3m = InstrumentFactory.create(
        'EuroOpt AMZN Listed 3M Call 3400 49.25 5 (AMZN_ATM_3M_CALL)'
    ).underlying(spx)
    print(spx_3m)
    risk_free_rate = 0.015
    heston_param = HestonProcessParam(
            spot=spx.get_quote(), 
            drift=spx.get_annual_yield(), 
            dividend=spx.get_dividend_yield(),
            spot_var=0.095078, kappa=6.649480, theta=0.391676, 
            rho=-0.796813, vov=0.880235
        )
    heston_process = create_heston_process(heston_param)
    gbm_param = GBMProcessParam(
            spot=spx.get_quote(), 
            drift=spx.get_annual_yield(), 
            dividend=spx.get_dividend_yield(), 
            vol=0.3
        )
    bsm_process = create_gbm_process(gbm_param)

    num_path = 10
    num_step = 90    
    step_days = 1
    step_size = time_from_days(step_days)
    spx.set_pricing_engine(heston_process, step_size, num_step)
    for i in range(num_path):
        for j in range(num_step+1):
            print("Days ", get_cur_days())
            spx_price, spx_vol = spx.get_price()
            print(spx.get_name(), spx_price, spx_vol)
            call_heston_process = create_heston_process(
                HestonProcessParam(
                    spot=spx_price, spot_var=min(1e-4, spx_vol), drift=risk_free_rate, dividend=heston_param.dividend,
                    kappa=heston_param.kappa, theta=heston_param.theta, rho=heston_param.rho, vov=heston_param.vov
                )
            )
            call_bsm_process = create_gbm_process(
                GBMProcessParam(
                    spot=spx_price, drift=risk_free_rate, dividend=heston_param.dividend, vol=spx_vol**0.5
                )
            )
            bsm_engine = ql.AnalyticEuropeanEngine(call_bsm_process)
            heston_engine = ql.AnalyticHestonEngine(ql.HestonModel(call_heston_process),0.01,1000)
            spx_3m.set_pricing_engine(heston_engine, bsm_engine)
            print(spx_3m.get_name(), spx_3m.get_price())
            move_days(step_days)
        reset_date()
        
    spx.set_pricing_engine(bsm_process, step_size, num_step)
    for i in range(num_path):
        for j in range(num_step+1):
            print(get_cur_days())
            spx_price, spx_vol = spx.get_price()
            print(spx.get_name(), spx_price, spx_vol)
            call_bsm_process = create_gbm_process(
                GBMProcessParam(
                    spot=spx_price, drift=risk_free_rate, dividend=gbm_param.dividend, vol=gbm_param.vol
                )
            )
            engine = ql.AnalyticEuropeanEngine(call_bsm_process)
            spx_3m.set_pricing_engine(engine)
            print(spx_3m.get_name(), spx_3m.get_price())
            move_days(step_days)
        reset_date()