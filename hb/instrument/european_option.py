import QuantLib as ql
from hb.instrument.instrument import Instrument
from hb.transaction_cost.transaction_cost import TransactionCost
from hb.utils.termstructure import *
from hb.utils.date import *
from hb.utils.consts import *
from hb.pricing import blackscholes
from hb.utils import consts
import numpy as np
import math
import tf_quant_finance as tff 


class EuropeanOption(Instrument):
    def __init__(self, name: str, option_type: str, strike: float, 
                 maturity: float, tradable: bool,
                 transaction_cost: TransactionCost = None,
                 underlying = None):
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if option_type=="Call" else ql.Option.Put, 
                                       strike)
        exercise = ql.EuropeanExercise(add_time(maturity))
        self._option = ql.VanillaOption(payoff, exercise)
        self._maturity_time = maturity
        self._call = option_type=="Call"
        self._strike = strike
        super().__init__(name=name, tradable=tradable, 
                         transaction_cost=transaction_cost, 
                         underlying=underlying)

    def set_simulator(self, simulator_handler, counter_handler):
        super().set_simulator(simulator_handler, counter_handler)
        ir = self._simulator_handler.get_obj().get_ir()
        spot = 100
        dividend = self._underlying.get_dividend_yield()
        vol = 0.2
        flat_ts = create_flat_forward_ts(ir)
        dividend_ts = create_flat_forward_ts(dividend)
        self._spot_handle = ql.RelinkableQuoteHandle(ql.SimpleQuote(spot))
        self._flat_vol_ts_handle = ql.RelinkableBlackVolTermStructureHandle(
            ql.BlackConstantVol(get_date(), calendar, vol, day_count)
        )
        bsm_process = ql.BlackScholesMertonProcess(self._spot_handle, 
                                                   dividend_ts, 
                                                   flat_ts, 
                                                   self._flat_vol_ts_handle)
        self._option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))

    def get_strike(self) -> float:
        return self._strike

    def get_is_call(self) -> bool:
        return self._call
    
    def get_is_expired(self) -> bool:
        """check if instrument expires

        Returns:
            bool: True if it expires, False otherwise
        """
        return (self._maturity_time-get_cur_time()) <= 1e-5

    def get_intrinsic_value(self):
        """The discounted intrinsic value

        Returns:
            float: The discounted intrinsic value
        """
        spot = self._underlying.get_price()
        tau_e = self.get_remaining_time()
        r = self._simulator_handler.get_obj().get_ir()
        q = self._underlying.get_dividend_yield()
        fwd = spot*math.exp((r-q)*tau_e)
        if self._call:
            intrinsic_value = 0. if fwd <= self._strike else fwd - self._strike
        else:
            intrinsic_value = 0. if fwd >= self._strike else self._strike - fwd
        return math.exp(-r*tau_e)*intrinsic_value

    def get_maturity_time(self):
        return self._maturity_time

    def get_remaining_time(self):
        return self._maturity_time - get_cur_time()

    def get_maturity_date(self):
        return date_from_time(self._maturity_time, ref_date=date0)

    def exercise(self) -> float:
        super().exercise()
        spot = self._underlying.get_price()
        if self._call:
            delivery = 0. if spot <= self._strike else 1.
        else:
            delivery = 0. if spot >= self._strike else -1.
        return delivery

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

    def get_is_physical_settle(self):
        """Assume Listed European Option is cash settled, and OTC is physical settled

        Returns:
            bool: option is physical settled or cash settled 
        """
        return not self._tradable

    def get_implied_vol(self, path_i:int=None, step_i:int=None) -> float:
        if (path_i is None) or (step_i is None):
            path_i, step_i = self._counter_handler.get_obj().get_path_step()
        imp_vol_surf = self._simulator_handler.get_obj().get_implied_vol_surface(self._underlying.get_name(), path_i, step_i)
        return imp_vol_surf.get_black_vol(t=self.get_remaining_time(),k=self._strike).numpy()

    def _get_price(self, path_i: int, step_i: int) -> float:
        if (abs(self._maturity_time-get_cur_time()) < 1e-5):
            # expiry, not exercised
            option_price = self.get_intrinsic_value()
        elif (self._maturity_time-get_cur_time() < 1e-5):
            # past expiry, or exercised
            option_price = 0.
        else:
            spot = float(self._underlying._get_price(path_i, step_i))
            vol = float(self.get_implied_vol(path_i, step_i))
            self._spot_handle.linkTo(ql.SimpleQuote(spot))
            self._flat_vol_ts_handle.linkTo(
                ql.BlackConstantVol(get_date(), calendar, vol, day_count)
            )
            option_price = self._option.NPV()
        return option_price

    def get_delta(self, path_i: int=None, step_i: int=None) -> float:
        if (abs(self._maturity_time-get_cur_time()) < 1e-5):
            # expiry, not exercised
            spot = self._underlying.get_price()
            if self._call:
                delta = 0. if spot <= self._strike else 1.
            else:
                delta = 0. if spot >= self._strike else -1.
            return delta
        elif (self._maturity_time-get_cur_time() < 1e-5):
            # past expiry, or exercised
            return 0.
        if (path_i is None) or (step_i is None):
            path_i, step_i = self._counter_handler.get_obj().get_path_step()
        spot = float(self._underlying._get_price(path_i, step_i))
        vol = float(self.get_implied_vol(path_i, step_i))
        self._spot_handle.linkTo(ql.SimpleQuote(spot))
        self._flat_vol_ts_handle.linkTo(
            ql.BlackConstantVol(get_date(), calendar, vol, day_count)
        )
        return self._option.delta()
        
    def get_gamma(self, path_i: int=None, step_i: int=None) -> float:
        if (abs(self._maturity_time-get_cur_time()) < 1e-5):
            # expiry, not exercised
            return 0
        elif (self._maturity_time-get_cur_time() < 1e-5):
            # past expiry, or exercised
            return 0.
        if (path_i is None) or (step_i is None):
            path_i, step_i = self._counter_handler.get_obj().get_path_step()
        spot = float(self._underlying._get_price(path_i, step_i))
        vol = float(self.get_implied_vol(path_i, step_i))
        self._spot_handle.linkTo(ql.SimpleQuote(spot))
        self._flat_vol_ts_handle.linkTo(
            ql.BlackConstantVol(get_date(), calendar, vol, day_count)
        )
        return self._option.gamma()

    def __repr__(self):
        return 'EuroOpt {underlying_name} {list_otc} {maturity} {call_put} {strike:.2f} {transaction_cost} ({name})' \
                .format(underlying_name=self._underlying.get_name(),
                        list_otc="Listed" if self._tradable else "OTC",
                        maturity=str_from_date(date_from_time(self._maturity_time)),
                        call_put="Call" if self._call else "Put",
                        strike=self._strike, 
                        transaction_cost=self._transaction_cost.get_percentage_cost()*100,
                        name=self._name)


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
