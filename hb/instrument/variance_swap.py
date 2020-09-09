from hb.instrument.instrument import Instrument
from hb.instrument.european_option import EuropeanOption
from hb.instrument.stock import Stock
from hb.transaction_cost.transaction_cost import TransactionCost
from hb.utils.date import *
from hb.utils import consts
import math
import numpy as np
import QuantLib as ql
import scipy.integrate as integrate


class VarianceSwap(Instrument):
    def __init__(self, name: str, strike: float, 
                 maturity: float, alpha: float=1.0, 
                 underlying: Stock = None):
        """Variance Swap
           For a variance swap, the environment needs be daily evolving, 
           where realised variance is calculated by daily stock price
        Args:
            name (str): name of the instrument
            strike (float): strike price
            maturity (float): maturity time
            alpha (float, optional): converting parameter $ per volatility-square. Defaults to 1.0.
            underlying (Stock, optional): underlying stock. Defaults to None.
        """
        self._maturity = maturity
        self._strike = strike
        self._alpha = alpha
        self._s_tm = None
        self._realized_var = 0.
        self._tm = None
        self._pred_ind = 0
        super().__init__(name, False, None, None, underlying, None)

    def get_strike(self) -> float:
        return self._strike

    def set_pricing_engine(self):
        pass

    def get_sim_price(self) -> float:
        """price of simulated episode at current time

        Returns:
            float: price
        """
        cur_time = get_cur_time()
        if cur_time < 1e-5: 
            # Time 0
            self._s_tm, _ = self._underlying.get_price()
            self._tm = 0.
            self._realized_var = 0.
        else:
            # update realised variance
            _s_t, _ = self._underlying.get_price()
            self._realized_var = 1/cur_time * (self._tm*self._realized_var + \
                                               (math.log(_s_t/self._s_tm))**2) 
            self._tm = cur_time
            self._s_tm = _s_t
        if abs(self._maturity - cur_time) < 1e-5:
            # expiry
            return self._alpha*(self._realized_var - self._strike)
        elif (self._maturity - cur_time) <= -1e-5:
            # past expiry
            return 0.
        else:
            # price contains two parts
            # realised variance and future variance expectation
            # future variance
            # cutoff $S^*=Fwd$
            s0, v0 = self._underlying.get_price()
            r = self._underlying.get_process_param().risk_free_rate
            tau = self.get_remaining_time()
            s_star = s0*math.exp(r*tau)
            integration = 0
            k_min = 0.4
            k_max = 1.6

            def integ_put(k):
                euro_opt = EuropeanOption("VarSwapPrice", "Put", k, 
                                          tau, False, reset_time=False) \
                                                  .underlying(self._underlying)
                return euro_opt.get_price()/k**2
            def integ_call(k):
                euro_opt = EuropeanOption("VarSwapPrice", "Call", k, 
                                            tau, False, reset_time=False) \
                                                .underlying(self._underlying)
                return euro_opt.get_price()/k**2
            put_integration, _ = integrate.quad(integ_put, 0., s_star)
            call_integration, _ = integrate.quad(integ_put, s_star, np.inf) 
            integration = put_integration + call_integration
            exp_var = 2*math.exp(r*tau)/tau*(integration)
            var_leg = 1/self._maturity* (cur_time*self._realized_var \
                                         + (self._maturity - cur_time)*exp_var)
            print(var_leg)
            return self._alpha*(var_leg - self._strike)

    def get_pred_price(self) -> float:
        """price of prediction episode at current time
        Returns:
            float: price
        """
        self._pred_ind += 1
        if self._cur_pred_path is None:
            self._cur_pred_path = -1
        if (abs(self._cur_price[0]-0.0) < 1e-5):
            # start of a new episode
            if self._cur_pred_path == self._pred_episodes:
                # run out all episodes, start repeating
                self._cur_pred_path = 0
            else:
                # continue next path
                if (self._cur_pred_file is None) and (self._cur_pred_path != -1):
                    self._pred_price_path += [[]]
                self._cur_pred_path += 1
            self._pred_ind = 0
        if self._cur_pred_file:
            # use loaded pred episodes
            price = self._pred_price_path[self._cur_pred_path][self._pred_ind]
        else:
            # simulate pred episodes
            price = self.get_sim_price()
            self._pred_price_path[self._cur_pred_path] = self._pred_price_path[self._cur_pred_path] + [price]
            if (self._pred_ind == self._num_steps) and \
                ((self._cur_pred_path + 1) == self._pred_episodes):
                # save pred episodes:
                self.save_pred_episodes(self._underlying._cur_pred_file)

        return price

    def get_maturity_time(self) -> float:
        """maturity time of the instrument from inception

        Returns:
            float: maturity time
        """
        return self._maturity

    def get_remaining_time(self) -> float:
        """remaining time of the instrument

        Returns:
            float: remaining time
        """
        return self._maturity - get_cur_time()

    def get_is_expired(self) -> bool:
        """check if instrument expires

        Returns:
            bool: True if it expires, False otherwise
        """
        return self.get_remaining_time() <= 1e-5

    def get_realised_var(self) -> float:
        """realised variance of the contract

        Returns:
            float: the realised variance in past
        """    
        return self._realized_var

    def __repr__(self):
        return f'Variance Swap {self._name}: \nunderlying=({str(self._underlying)})\nMaturity={get_period_str_from_time(self._maturity)}, VarStrike={self._strike}, VarNotional={self._alpha}'


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
    spx_var_swap = InstrumentFactory.create(
        'VarSwap AMZN 3M 0.04 1 (AMZN_VAR_SWAP)'
    ).underlying(spx)
    print(spx_var_swap)
    risk_free_rate = 0.015
    heston_param = HestonProcessParam(
            risk_free_rate=0.015,
            spot=spx.get_quote(), 
            drift=spx.get_annual_yield(), 
            dividend=spx.get_dividend_yield(),
            spot_var=0.096024, kappa=6.288453, theta=0.397888, 
            rho=-0.696611, vov=0.753137, use_risk_free=False
        )
    
    num_path = 1
    num_step = 90    
    step_days = 1
    step_size = time_from_days(step_days)
    spx.set_num_steps(num_step)
    spx.set_pricing_engine(step_size, heston_param)
    heston_prices = np.zeros([num_path, num_step])
    times = np.zeros([num_step])
    
    for i in range(num_path):
        for j in range(num_step):
            times[j] = get_cur_days()
            # print("Days ", get_cur_days())
            heston_prices[i][j] = spx_var_swap.get_price()
            move_days(step_days)
        reset_date()

    import matplotlib.pyplot as plt

    for i in range(num_path):
        plt.plot(times, heston_prices[i, :], lw=0.8, alpha=0.6)
    plt.title("Heston Simulation Variance Swap Price")
    plt.show()
    print(heston_prices[:,-1].mean())
    

