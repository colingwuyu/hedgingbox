from hb.instrument.instrument import Instrument
from hb.instrument.european_option import EuropeanOption
from hb.instrument.stock import Stock
from hb.transaction_cost.transaction_cost import TransactionCost
from hb.utils.date import *
from hb.utils import consts
from hb.utils.process import *
import math
import numpy as np
import QuantLib as ql
import scipy.integrate as integrate
from typing import List


class VarianceSwap(Instrument):
    def __init__(self, name: str, vol_strike: float, 
                 maturity: float, alpha: float=1.0, notional: float=100.,
                 underlying: Stock = None,
                 replicating_portfolio: List[EuropeanOption] = [],
                 dk: float = 5.):
        """Variance Swap
           For a variance swap, the environment needs be daily evolving, 
           where realised variance is calculated by daily stock price
           Using replicating portfolio to price variance swap
        Args:
            name (str): name of the instrument
            vol_strike (float): volatility strike price
            maturity (float): maturity time
            alpha (float, optional): converting parameter $ per volatility-square. Defaults to 1.0.
            notional (float, optional): notional of the contract. Defaults to 100.0.
            underlying (Stock, optional): underlying stock. Defaults to None.
            replicating_portfolio (List[EuropeanOption], optional): the european options used to price the variance swap by replicating portfolio. Default to empty.
            dk (float, optional): use as the boundary case delta k for the european options' weights calculation. 
        """
        self._maturity = maturity
        self._strike = vol_strike**2
        self._alpha = alpha
        self._notional = notional
        self._s_tm = None
        self._realized_var = 0.
        self._tm = None
        self._pred_ind = 0
        self._dk = dk
        self._puts = np.array([])
        self._put_strikes = np.array([])
        self._calls = np.array([])
        self._call_strikes = np.array([])
        if len(replicating_portfolio) != 0:
            for euro_opt in replicating_portfolio:
                if euro_opt.get_maturity_time() != self._maturity:
                    # hedging options need have the same maturity
                    continue
                if euro_opt.get_is_call() and (euro_opt.get_strike() not in self._call_strikes):
                    self._calls = np.append(self._calls, euro_opt)
                    self._call_strikes = np.append(self._call_strikes, euro_opt.get_strike())
                elif euro_opt.get_strike() not in self._put_strikes:
                    self._puts = np.append(self._puts, euro_opt)
                    self._put_strikes = np.append(self._put_strikes, euro_opt.get_strike())
            call_order = np.argsort(self._call_strikes)
            put_order = np.argsort(self._put_strikes)[::-1]
            self._calls = self._calls[call_order]
            self._call_strikes = self._call_strikes[call_order]
            self._puts = self._puts[put_order]
            self._put_strikes = self._put_strikes[put_order]
            # assure the strike boundary is same for call and put options
            assert abs(self._call_strikes[0]-self._put_strikes[0]) < 1e-5
            self._f = self._call_strikes[0]
            self._call_weights = self._compute_option_weights(self._call_strikes, True)
            self._put_weights = self._compute_option_weights(self._put_strikes, False)
            self._option_weights = np.append(self._put_weights, self._call_weights)
            self._options = np.append(self._puts, self._calls)
        super().__init__(name, False, None, None, underlying, None)

    def _compute_option_weights(self, strikes: List[float], is_call: bool) -> List[float]:
        option_weights = np.zeros(len(strikes))
        if len(strikes) == 0:
            return option_weights
        strikes = self._call_strikes if is_call else self._put_strikes
        if is_call:
            strikes = np.append(strikes, strikes[-1]+self._dk)
        else:
            strikes = np.append(strikes, max(strikes[-1]-self._dk, 0.))
        slope = 0.; prev_slope = 0.
        for strike_i in range(len(strikes)-1):
            # slope = abs(self._comput_log_payoff(strikes[strike_i+1]) - 
            #             self._comput_log_payoff(strikes[strike_i])  
            #             / (strikes[strike_i+1] - strikes[strike_i]))
            # if strike_i == 0:
            #     option_weights[strike_i] = slope
            # else:
            #     option_weights[strike_i] = slope - prev_slope
            option_weights[strike_i] = abs(1/strikes[strike_i+1] - 1/strikes[strike_i])*2./self.get_remaining_time()
            prev_slope = slope
        return option_weights

    def _comput_log_payoff(self, strike: float) -> float:
        return (2./self.get_remaining_time())*((strike - self._f)/self._f - math.log(strike/self._f))

    def _compute_replication_portfolio(self, option_weights: List[float], options: List[EuropeanOption]) -> float:
        options_value = 0.
        spot, _ = self._underlying.get_price()
        r = self._underlying.get_risk_free_rate()
        df = math.exp(-r*self.get_remaining_time())
        fwd = spot/df
        process_param = self._underlying.get_process_param()
        for option_i in range(len(option_weights)):
            # self._param = GBMProcessParam(
            #         risk_free_rate=process_param.risk_free_rate, spot=spot, 
            #         drift=process_param.risk_free_rate, 
            #         dividend=self._underlying.get_dividend_yield(), 
            #         vol=options[option_i].get_quote(), use_risk_free=True
            #     )
            # bsm_process = create_gbm_process(self._param)        
            # options[option_i]._option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
            # option_price=options[option_i]._option.NPV()
            option_price = options[option_i].get_price()
            weight = option_weights[option_i]
            options_value += option_price*weight
        ret_var = 2. * r \
                    - 2./self.get_remaining_time() * \
                        ((fwd - self._f)/self._f + 
                         math.log(self._f/spot)) \
                    + options_value / df
        return ret_var

    def replicating_opts(self, repliacting_portfolio: List[EuropeanOption]):
        self._puts = np.array([])
        self._put_strikes = np.array([])
        self._calls = np.array([])
        self._call_strikes = np.array([])
        for euro_opt in repliacting_portfolio:
            if euro_opt.get_maturity_time() != self._maturity:
                # hedging options need have the same maturity
                continue
            if euro_opt.get_is_call() and (euro_opt.get_strike() not in self._call_strikes):
                self._calls = np.append(self._calls, euro_opt)
                self._call_strikes = np.append(self._call_strikes, euro_opt.get_strike())
            elif euro_opt.get_strike() not in self._put_strikes:
                self._puts = np.append(self._puts, euro_opt)
                self._put_strikes = np.append(self._put_strikes, euro_opt.get_strike())
        call_order = np.argsort(self._call_strikes)
        put_order = np.argsort(self._put_strikes)[::-1]
        self._calls = self._calls[call_order]
        self._call_strikes = self._call_strikes[call_order]
        self._puts = self._puts[put_order]
        self._put_strikes = self._put_strikes[put_order]
        # assure the strike boundary is same for call and put options
        assert abs(self._call_strikes[0]-self._put_strikes[0]) < 1e-5
        self._f = self._call_strikes[0]
        self._call_weights = self._compute_option_weights(self._call_strikes, True)
        self._put_weights = self._compute_option_weights(self._put_strikes, False)
        self._option_weights = np.append(self._call_weights, self._put_weights)
        self._options = np.append(self._calls, self._puts)
        return self

    def get_strike(self) -> float:
        return self._strike

    def set_pricing_engine(self):
        pass

    def get_hedging_instrument_names(self):
        hedging_names = [self._underlying.get_name()]
        for opt in self._options:
            hedging_names += [opt.get_name()]
        return hedging_names

    def get_greeks(self):
        """Return greeks
            A long position of Variance swap has greeks:
                - underlying share: -1
                - replicating options: option_weight
        """
        hedging_ratios = {self._underlying.get_name(): -2./self._maturity/self._f}
        for opt, weight in zip(self._options, 2./self._maturity*self._option_weights):
            hedging_ratios[opt.get_name()] = weight
        return hedging_ratios
        
    def exercise(self) -> float:
        """ update the status to exercise. and returns the hedging position of underlying stocks

        Returns:
            float: [description]
        """
        super().exercise()
        spot, _ = self._underlying.get_price()
        delivery = -1.
        return delivery

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
            return self._alpha*self._notional*(self._realized_var - self._strike)
        elif (self._maturity - cur_time) <= -1e-5:
            # past expiry
            return 0.
        else:
            # price contains two parts
            # realised variance and future variance expectation
            # future variance
            r = self._underlying.get_risk_free_rate()
            df = math.exp(-r*self.get_remaining_time())
            fut_variance = self._compute_replication_portfolio(self._option_weights, self._options)
            variance = cur_time*self._realized_var + self.get_remaining_time()*fut_variance
            return df*self._alpha*self._notional*(variance/self._maturity - self._strike) 

    def get_is_physical_settle(self):
        return False

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
    

