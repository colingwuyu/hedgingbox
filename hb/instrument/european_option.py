import QuantLib as ql
from hb.instrument.instrument import Instrument
from hb.transaction_cost.transaction_cost import TransactionCost
from hb.utils.process import *
from hb.utils.date import *
from hb.pricing import blackscholes
import numpy as np

class EuropeanOption(Instrument):
    def __init__(self, name: str, option_type: str, strike: float, 
                 maturity: float, tradable: bool, quote: float = None,
                 transaction_cost: TransactionCost = None,
                 underlying: Instrument = None, trading_limit: float = 1e10):
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if option_type=="Call" else ql.Option.Put, 
                                       strike)
        exercise = ql.EuropeanExercise(add_time(maturity))
        self._option = ql.VanillaOption(payoff, exercise)
        self._maturity_time = maturity
        self._call = option_type=="Call"
        self._strike = strike
        self._back_up_pricing_engine = None
        self._cur_time = None
        self._param = None
        super().__init__(name, tradable, quote, transaction_cost, underlying, trading_limit)

    def get_quote(self) -> float:
        return self._quote
    
    def set_quote(self, quote: float):
        self._quote = quote
    
    def quote(self, quote: float):
        self._quote = quote
        return self

    def get_strike(self) -> float:
        return self._strike

    def set_pricing_engine(self):
        process_param = self._underlying.get_process_param()
        underlyer_price, underlyer_var = self._underlying.get_price()
        if isinstance(process_param, HestonProcessParam):
            # Heston model
            self._param = HestonProcessParam(
                    risk_free_rate=process_param.risk_free_rate,spot=underlyer_price, spot_var=min(1e-4, underlyer_var), 
                    drift=process_param.risk_free_rate, dividend=self._underlying.get_dividend_yield(),
                    kappa=process_param.kappa, theta=process_param.theta, 
                    rho=process_param.rho, vov=process_param.vov, use_risk_free=True
                )
            heston_process = create_heston_process(self._param)
            bsm_process = create_gbm_process(
                GBMProcessParam(
                    risk_free_rate=process_param.risk_free_rate, spot=underlyer_price, 
                    drift=process_param.risk_free_rate,
                    dividend=self._underlying.get_dividend_yield(), 
                    vol=underlyer_var**0.5, use_risk_free=True
                )
            )
            self._option.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(heston_process),0.01,1000))
            self._back_up_pricing_engine = ql.AnalyticEuropeanEngine(bsm_process)
        elif isinstance(process_param, GBMProcessParam):
            # BSM Model
            self._param = GBMProcessParam(
                    risk_free_rate=process_param.risk_free_rate, spot=underlyer_price, 
                    drift=process_param.risk_free_rate, 
                    dividend=self._underlying.get_dividend_yield(), 
                    vol=process_param.vol, use_risk_free=True
                )
            bsm_process = create_gbm_process(self._param)        
            self._option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))

    def get_is_expired(self) -> bool:
        """check if instrument expires

        Returns:
            bool: True if it expires, False otherwise
        """
        return (self._maturity_time-get_cur_time()) <= 1e-5

    def get_price(self):
        if (self._cur_time != get_cur_time()):
            self.set_pricing_engine()
            self._cur_time = get_cur_time()
        if abs(self._maturity_time-get_cur_time()) < 1e-5:
            # expiry
            return self.get_intrinsic_value()
        elif self._maturity_time-get_cur_time() <= -1e-5:
            # past expiry
            return 0.
        else:
            # price before expiry
            try:
                price = self._option.NPV()
                if price - self.get_intrinsic_value() < -1e-5:
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

    def get_remaining_time(self):
        return self._maturity_time - get_cur_time()

    def get_maturity_date(self):
        return date_from_time(self._maturity_time, ref_date=date0)

    def get_delivery_amount(self):
        assert abs(self.get_remaining_time()) < 1e-5
        if self._call:
            return 1.
        else:
            return -1.
    
    def get_receive_amount(self):
        assert abs(self.get_remaining_time()) < 1e-5
        if self._call:
            return self._strike
        else:
            return -self._strike

    def get_is_exercised(self):
        assert abs(self.get_remaining_time()) < 1e-5
        spot, _ = self._underlying.get_price()
        if self._call:
            return False if spot <= self._strike else True
        else:
            return False if spot >= self._strike else True

    def get_is_physical_settle(self):
        return True

    def get_implied_vol(self) -> float:
        if isinstance(self._param, GBMProcessParam):
            return self._param.vol
        else:
            return blackscholes.implied_vol(
                self._call, self._strike, self.get_remaining_time(),
                self.get_price(),
                GBMProcessParam(
                                risk_free_rate=self._param.risk_free_rate, 
                                spot=self._param.spot,
                                drift=self._param.drift, dividend=self._param.dividend, 
                                vol=0.2,
                                use_risk_free=True
                               )
            )

    def _get_gbm_param(self) -> GBMProcessParam:
        if isinstance(self._param, GBMProcessParam):
            return self._param
        else:
            return GBMProcessParam(
                                risk_free_rate=self._param.risk_free_rate, 
                                spot=self._param.spot,
                                drift=self._param.drift, dividend=self._param.dividend, 
                                vol=self.get_implied_vol(),
                                use_risk_free=True
                            )

    def get_delta(self) -> float:
        """BS Delta

        Returns:
            float: BlackScholes Model Delta
        """
        if isinstance(self._param, GBMProcessParam):
            delta_value = self._option.delta()
            if np.isnan(delta_value):
                delta_value = blackscholes.delta_bk(True, self._param.spot, self._param.risk_free_rate, 
                                                    self._param.dividend, self._strike, 
                                                    self._param.vol, self.get_remaining_time(), 
                                                    self.get_remaining_time())
            return delta_value
        else:
            return blackscholes.delta(self._call, self._strike,
                                    self._get_gbm_param(),
                                    self._maturity_time-self._cur_time)

    def get_gamma(self) -> float:
        """BS Gamma

        Returns:
            float: BlackScholes Model Gamma
        """
        if isinstance(self._param, GBMProcessParam):
            return self._option.gamma()
        else:
            return blackscholes.gamma(self._call, self._strike,
                                    self._get_gbm_param(),
                                    self._maturity_time-self._cur_time)

    def get_vega(self) -> float:
        """BS Vega

        Returns:
            float: BlackScholes Model Vega
        """
        if isinstance(self._param, GBMProcessParam):
            return self._option.vega()
        else:
            return blackscholes.vega(self._call, self._strike,
                                    self._get_gbm_param(),
                                    self._maturity_time-self._cur_time)

    def __repr__(self):
        return f'European Option {self._name}: \nunderlying=({str(self._underlying)})\noption_type={self._call}, maturity={get_period_str_from_time(self._maturity_time)}, tradable={self._tradable}, iv={self._quote}, transaction_cost={str(self._transaction_cost)}'


if __name__ == "__main__":
    from hb.transaction_cost.percentage_transaction_cost import PercentageTransactionCost
    from hb.instrument.instrument_factory import InstrumentFactory
    from hb.utils.date import *
    from hb.utils.consts import *
    from hb.utils.process import *
    import numpy as np
    
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
            risk_free_rate=0.015,
            spot=spx.get_quote(), 
            drift=spx.get_annual_yield(), 
            dividend=spx.get_dividend_yield(),
            spot_var=0.096024, kappa=6.288453, theta=0.397888, 
            rho=-0.696611, vov=0.753137, use_risk_free=False
        )
    gbm_param = GBMProcessParam(
            risk_free_rate = 0.015,
            spot=spx.get_quote(), 
            drift=spx.get_annual_yield(), 
            dividend=spx.get_dividend_yield(), 
            vol=0.5,
            use_risk_free=False
        )
    
    num_path = 1_000
    num_step = 90    
    step_days = 1
    step_size = time_from_days(step_days)
    spx.set_pricing_engine(step_size, num_step, heston_param)
    heston_prices = np.zeros([num_path, num_step])
    times = np.zeros([num_step])
    
    for i in range(num_path):
        for j in range(num_step):
            times[j] = get_cur_days()
            # print("Days ", get_cur_days())
            heston_prices[i][j] = spx_3m.get_price()
            move_days(step_days)
        reset_date()
        
    spx.set_pricing_engine(step_size, num_step, gbm_param)
    gbm_prices = np.zeros([num_path, num_step])
    for i in range(num_path):
        for j in range(num_step):
            # print(get_cur_days())
            gbm_prices[i][j] = spx_3m.get_price()
            move_days(step_days)
        reset_date()

    import matplotlib.pyplot as plt

    for i in range(num_path):
        plt.plot(times, heston_prices[i, :], lw=0.8, alpha=0.6)
    plt.title("Heston Simulation Option")
    plt.show()
    print(heston_prices[:,-1].mean())
    for i in range(num_path):
        plt.plot(times, gbm_prices[i, :], lw=0.8, alpha=0.6)
    plt.title("GBM Simulation")
    plt.show()
    print(gbm_prices[:,-1].mean())
