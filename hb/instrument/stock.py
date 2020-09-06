import QuantLib as ql
from hb.instrument.instrument import Instrument
from hb.transaction_cost.transaction_cost import TransactionCost
from hb.utils.date import *
from hb.utils.process import *
from typing import Union
import numpy as np


class Stock(Instrument):
    def __init__(self, name: str, quote: float, 
                 annual_yield: float,
                 dividend_yield: float,
                 transaction_cost: TransactionCost,
                 pred_episodes=1_000):
        """Stock

        Args:
            name ([str]): stock name (ticker)
            quote ([float]): price
            annual_yield ([float]): annual return
            dividend_yield ([float]): annual dividend yield
            transaction_cost ([TransactionCost]): transaction_cost calculator 
        """
        self._spot_path = None
        self._dividend_yield = dividend_yield
        self._annual_yield = annual_yield
        self._vol_path = None
        self._repeat_episodes = None
        self._length = None
        self._num_step = None
        self._step_size = None
        self._cur_path = None
        self._stoc_vol = None
        self._process_param = None
        self._pred_mode = False
        self._pred_episodes = pred_episodes
        self._pred_spot_path = None
        self._pred_vol_path = None
        self._cur_pred_path = 0
        super().__init__(name, True, quote, transaction_cost)

    def get_dividend_yield(self) -> float:
        return self._dividend_yield

    def set_dividend_yield(self, dividend_yield):
        self._dividend_yield = dividend_yield

    def dividend_yield(self, dividend_yield):
        self._dividend_yield = dividend_yield
        return self
    
    def get_annual_yield(self) -> float:
        return self._annual_yield

    def set_annual_yield(self, annual_yield):
        self._annual_yield = annual_yield

    def annual_yield(self, annual_yield):
        self._annual_yield = annual_yield
        return self

    def get_quote(self) -> float:
        return self._quote
    
    def set_quote(self, quote: float):
        self._quote = quote
    
    def quote(self, quote: float):
        self._quote = quote
        return self

    def get_market_value(self, holding):
        price, _ = self.get_price()
        return price*holding

    def set_process_param(self, process_param):
        self._process_param = process_param

    def get_process_param(self):
        return self._process_param

    def set_pred_episodes(self, pred_episodes: int):
        self._pred_episodes = pred_episodes

    def set_pricing_engine(self, step_size, num_step, 
                           pricing_engine: Union[GBMProcessParam, HestonProcessParam]=None, repeat_episodes=None):
        if pricing_engine is not None:
            self._process_param = pricing_engine
        self._process = create_process(self._process_param)
        self._repeat_episodes = repeat_episodes
        self._cur_path = None
        self._step_size = step_size
        self._num_step = num_step
        self._length = num_step*step_size
        times = ql.TimeGrid(self._length, self._num_step)
        dimension = self._process.factors()
        if dimension == 1:
            self._stoc_vol = False
        elif dimension == 2:
            self._stoc_vol = True
        rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(dimension * self._num_step, ql.UniformRandomGenerator()))
        self._pricing_engine = ql.GaussianMultiPathGenerator(self._process, list(times), rng, False)

    def set_pred_mode(self, pred_mode: bool):
        self._pred_mode = pred_mode
        self._cur_pred_path = None

    def get_price(self):
        cur_time = get_cur_time()
        if (self._repeat_episodes is None) and (not self._pred_mode):
            # no repeat path
            self._cur_path = 0
            _cur_path = 0
            if (abs(cur_time-0.0) < 1e-5):
                # start a new path
                self._simulate()
        else: 
            # repeat path
            if self._pred_mode:
                _cur_path = self._cur_pred_path
                _repeat_episodes = self._pred_episodes
            else:
                _cur_path = self._cur_path
                _repeat_episodes = self._repeat_episodes
            if _cur_path is None:
                # simulate paths
                _cur_path = 0
                self._simulate()
            elif (abs(cur_time-0.0) < 1e-5):
                if (_cur_path + 1) == _repeat_episodes:
                    # repeat
                    _cur_path = 0
                else:
                    # continue next path
                    _cur_path += 1
        ind = int(round(cur_time/self._step_size))
        if self._pred_mode:
            spot = self._pred_spot_path[_cur_path][ind]
            self._cur_pred_path = _cur_path
        else:
            spot = self._spot_path[_cur_path][ind]
            self._cur_path = _cur_path
        if self._stoc_vol:
            # stochastic vol
            if self._pred_mode:
                vol = self._pred_vol_path[_cur_path][ind]
            else:
                vol = self._vol_path[_cur_path][ind]
        else:
            vol = 0. 
        return spot, vol

    def _simulate(self):
        _spot_path = []
        _vol_path = []
        if self._pred_mode:
            num_path = self._pred_episodes
        elif self._repeat_episodes is not None:
            num_path = self._repeat_episodes
        else:
            num_path = 1
        for i in range(num_path):
            sample_path = self._pricing_engine.next()
            values = sample_path.value()
            if self._stoc_vol:
                spot, vol = values
            else:
                spot = values
            if self._stoc_vol:
                _spot_path.append([x for x in spot])
                _vol_path.append([x for x in vol])
            else:
                _spot_path.append([x for x in spot[0]])
        
        if self._pred_mode:
            self._pred_spot_path = np.array(_spot_path) 
            self._pred_vol_path = np.array(_vol_path)
        else:
            self._spot_path = np.array(_spot_path)
            self._vol_path = np.array(_vol_path)


    def get_maturity_time(self):
        return 0.0

    def get_remaining_time(self):
        return 9999999

    def get_delivery_amount(self):
        return 0

    def __repr__(self):
        return f'Stock {self._name}: \nquote={self._quote}, annual_yield={self._annual_yield}, dividend_yield={self._dividend_yield}, transaction_cost={str(self._transaction_cost)}'

if __name__ == "__main__":
    from hb.transaction_cost.percentage_transaction_cost import PercentageTransactionCost
    from hb.utils.date import *
    from hb.utils.consts import *
    from hb.utils.process import *
    spx = Stock(name='AMZN', 
                quote=3400, 
                annual_yield=0.25,
                dividend_yield=0.0,
                transaction_cost = PercentageTransactionCost(0.001))
    print(spx)
    heston_param = HestonProcessParam(
            risk_free_rate=0.015,
            spot=spx.get_quote(), 
            drift=spx.get_annual_yield(), 
            dividend=spx.get_dividend_yield(),
            spot_var=0.096024, kappa=6.288453, theta=0.397888, 
            rho=-0.696611, vov=0.753137, use_risk_free=False
        )
    bsm_param = GBMProcessParam(
            risk_free_rate = 0.015,
            spot=spx.get_quote(), 
            drift=spx.get_annual_yield(), 
            dividend=spx.get_dividend_yield(), 
            vol=0.5, use_risk_free=False
        )
    num_path = 1_000
    num_step = 90    
    step_days = 1
    step_size = time_from_days(step_days)
    spx.set_pricing_engine(step_size, num_step, heston_param)
    spx.set_pred_episodes(2)
    heston_prices = np.zeros([num_path, num_step])
    heston_vars = np.zeros([num_path, num_step])
    times = np.zeros([num_step])
    for i in range(num_path):
        for j in range(num_step):
            times[j] = get_cur_days()
            # print(get_cur_days())
            price, vol = spx.get_price()
            heston_prices[i][j] = price
            heston_vars[i][j] = vol
            # print(spx.get_name(), spx.get_price())
            move_days(step_days)
        reset_date()

    # spx.set_pred_mode(pred_mode=True)
    # for i in range(num_path):
    #     for j in range(num_step):
    #         # print(get_cur_days())
    #         # print(spx.get_name(), spx.get_price())
    #         move_days(step_days)
    #     reset_date()
    # spx.set_pred_mode(pred_mode=False)

    spx.set_pricing_engine(step_size, num_step, bsm_param)
    gbm_prices = np.zeros([num_path, num_step])
    for i in range(num_path):
        for j in range(num_step):
            # print(get_cur_days())
            # print(spx.get_name(), spx.get_price())
            price, vol = spx.get_price()
            gbm_prices[i][j] = price
            move_days(step_days)
        reset_date()

    import matplotlib.pyplot as plt

    for i in range(num_path):
        plt.plot(times, heston_prices[i, :], lw=0.8, alpha=0.6)
    plt.title("Heston Simulation Spot")
    plt.show()
    for i in range(num_path):
        plt.plot(times, heston_vars[i, :], lw=0.8, alpha=0.6)
    plt.title("Heston Simulation Var")
    plt.show()
    print(heston_vars[:,-1].mean())
    print(heston_prices[:,-1].mean())
    for i in range(num_path):
        plt.plot(times, gbm_prices[i, :], lw=0.8, alpha=0.6)
    plt.title("GBM Simulation")
    plt.show()
    print(gbm_prices[:,-1].mean())
