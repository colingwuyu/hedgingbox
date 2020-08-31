from hb.instrument.instrument import Instrument
from hb.transaction_cost.transaction_cost import TransactionCost
from hb.utils.date import get_cur_time
import QuantLib as ql
import numpy as np


class Stock(Instrument):
    def __init__(self, name: str, quote: float, 
                 annual_yield: float,
                 dividend_yield: float,
                 transaction_cost: TransactionCost):
        """Stock

        Args:
            process ([ql.Process])
            transaction_cost ([TransactionCost]): [description]
        """
        self._spot_path = None
        self._dividend_yield = dividend_yield
        self._annual_yield = annual_yield
        self._vol_path = None
        self._repeat_path = None
        self._length = None
        self._step_size = None
        self._cur_path = None
        self._stoc_vol = None
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

    def set_pricing_engine(self, pricing_engine, step_size, num_step, repeat_path = None):
        self._repeat_path = repeat_path
        self._cur_path = None
        self._step_size = step_size
        self._length = num_step*step_size
        times = ql.TimeGrid(self._length, num_step)
        dimension = pricing_engine.factors()
        if dimension == 1:
            self._stoc_vol = False
        elif dimension == 2:
            self._stoc_vol = True
        rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(dimension * num_step, ql.UniformRandomGenerator()))
        self._pricing_engine = ql.GaussianMultiPathGenerator(pricing_engine, list(times), rng, False)

    def get_price(self):
        cur_time = get_cur_time()
        if self._repeat_path is None:
            # no repeat path
            self._cur_path = 0
            if (abs(cur_time-0.0) < 1e-5):
                # start a new path
                self._simulate(1)
        else: 
            # repeat path
            if self._cur_path is None:
                # simulate paths
                self._cur_path = 0
                self._simulate(self._repeat_path)
            elif (abs(cur_time-0.0) < 1e-5):
                if (self._cur_path + 1) == self._repeat_path:
                    # repeat
                    self._cur_path = 0
                else:
                    # continue next path
                    self._cur_path += 1
        ind = int(round(cur_time/self._step_size))
        spot = self._spot_path[self._cur_path][ind]
        if self._stoc_vol:
            # stochastic vol
            vol = self._vol_path[self._cur_path][ind]
        else:
            vol = 0. 
        return spot, vol

    def _simulate(self, num_path):
        self._spot_path = []
        self._vol_path = []

        for i in range(num_path):
            sample_path = self._pricing_engine.next()
            values = sample_path.value()
            if self._stoc_vol:
                spot, vol = values
            else:
                spot = values
            if self._stoc_vol:
                self._spot_path.append([x for x in spot])
                self._vol_path.append([x for x in vol])
            else:
                self._spot_path.append([x for x in spot[0]])
        
        self._spot_path = np.array(self._spot_path)
        self._vol_path = np.array(self._vol_path)


    def get_maturity_time(self):
        return 0.0

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
    heston_process = create_heston_process(
        HestonProcessParam(
            spot=spx.get_quote(), 
            drift=spx.get_annual_yield(), 
            dividend=spx.get_dividend_yield(),
            spot_var=0.095078, kappa=6.649480, theta=0.391676, 
            rho=-0.796813, vov=0.880235
        )
    )
    bsm_process = create_gbm_process(
        GBMProcessParam(
            spot=spx.get_quote(), 
            drift=spx.get_annual_yield(), 
            dividend=spx.get_dividend_yield(), 
            vol=0.3
        )
    )

    num_path = 3
    num_step = 10    
    step_days = 1
    step_size = time_from_days(step_days)
    spx.set_pricing_engine(heston_process, step_size, num_step)
    for i in range(num_path):
        for j in range(num_step):
            print(get_cur_days())
            print(spx.get_name(), spx.get_price())
            move_days(step_days)
        reset_date()

    spx.set_pricing_engine(bsm_process, step_size, num_step, 2)

    for i in range(num_path):
        for j in range(num_step):
            print(get_cur_days())
            print(spx.get_name(), spx.get_price())
            move_days(step_days)
        reset_date()
    

