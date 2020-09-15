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
                 maturity: float, var_notional: float=None, vega_notional: float=None,
                 underlying: Stock = None,
                 replicating_portfolio: List[EuropeanOption] = [],
                 dk: float = 5.):
        """Variance Swap
           For a variance swap, the environment needs be daily evolving, 
           where realised variance is calculated by daily stock price
           Using replicating portfolio to price variance swap
        Args:
            name (str): name of the instrument
            vol_strike (float): volatility strike (in terms of 100, for example implied vol = 0.25, then strike = 25, var_strike = 25**2)
            maturity (float): maturity time
            notional (float, optional): notional of the contract. Defaults to 1000.0.
            underlying (Stock, optional): underlying stock. Defaults to None.
            replicating_portfolio (List[EuropeanOption], optional): the european options used to price the variance swap by replicating portfolio. Default to empty.
            dk (float, optional): use as the boundary case delta k for the european options' weights calculation. 
        """
        self._pricing_method = 'Replicating'
        self._excl_realized_var = False
        self._maturity = maturity
        self._var_strike = vol_strike**2
        if var_notional:
            self._var_notional = var_notional
        else:
            self._var_notional = vega_notional/2/vol_strike
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

    def replicating_opts(self, repliacting_portfolio: List[EuropeanOption]):
        """Add a portfolio of OTM options for replicating variance swap
            Calculates the replicating weights of OTM options
            Derman's Method (Demeterfi et al., 1999)
            Replicating the constant dollar gamma payoff:
                $$f(x)=\frac{x}{K_0}-1-ln(\frac{x}{K_0})$$
        Args:
            repliacting_portfolio (List[EuropeanOption]): a list OTM puts and calls

        Returns:
            VarianceSwap: return self
        """
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
        
    def _compute_option_weights(self, strikes: List[float], is_call: bool) -> List[float]:
        """Compute the replicating weights of OTM options
            Derman's Method (Demeterfi et al., 1999)

        Args:
            strikes (List[float]): a list of OTM strikes call or put
            is_call (bool): True - OTM Calls; False - OTM Puts

        Returns:
            List[float]: the replicating weights of the input options
        """
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
            slope = abs((self._compute_log_payoff(strikes[strike_i+1]) - 
                        self._compute_log_payoff(strikes[strike_i])) / 
                        (strikes[strike_i+1] - strikes[strike_i]))
            if strike_i == 0:
                option_weights[strike_i] = slope
            else:
                option_weights[strike_i] = slope - prev_slope
            prev_slope = slope
        return option_weights

    def _compute_log_payoff(self, strike: float) -> float:
        """Log Payoff
                $$f(x)=\frac{x}{K_0}-1-ln(\frac{x}{K_0})$$

        Args:
            strike (float): strike x

        Returns:
            float: log payoff
        """
        return strike/self._f - 1 - math.log(strike/self._f)

    def _compute_replication_portfolio(self, option_weights: List[float], options: List[EuropeanOption]) -> float:
        options_value = 0.
        spot, _ = self._underlying.get_price()
        r = self._underlying.get_risk_free_rate()
        df = math.exp(-r*self.get_remaining_time())
        fwd = spot/df
        process_param = self._underlying.get_process_param()
        for option_i in range(len(option_weights)):
            option_price = options[option_i].get_price()
            weight = option_weights[option_i]
            options_value += option_price*weight
        ret_var = 2. * r \
                    - 2./self.get_remaining_time() * \
                        ((fwd - self._f)/self._f + 
                         math.log(self._f/spot)) \
                    + 2./self.get_remaining_time() * (options_value / df)
        return ret_var

    def _heston_fut_var(self):
        """Future variance based on Heston model
           The variance sde of Heston model follows CIR process 
            $$dv=\kappa (\theta - v)dt + \sigma \sqrt{v} dW_v$$
           The expected value of total variance
            $$E[\int_t^T{\sigma^2_t}dt]$$
           can be solved in closed form
            $$\theta+\frac{v_0-\theta}{\kappa T}(1-e^{-\kappa T})$$
        """
        tau = self.get_remaining_time()
        return (self._param.theta + 
                 (self._param.spot_var-self._param.theta)/self._param.kappa/tau 
                 * (1- math.exp(-self._param.kappa*tau)))

    def get_strike(self) -> float:
        return self._var_strike

    def set_pricing_engine(self):
        pass

    def get_hedging_instrument_names(self):
        hedging_names = [self._underlying.get_name()]
        for opt in self._options:
            hedging_names += [opt.get_name()]
        return hedging_names

    def get_var_notional(self):
        return self._var_notional

    def get_vega_notional(self):
        return self._var_notional*2*self._var_strike**0.5

    def set_pricing_method(self, pricing_method: str):
        """Pricing method

        Args:
            pricing_method (str): 'Replicating' (default in constructor) or 'Heston'
        """
        self._pricing_method = pricing_method

    def set_excl_realized_var(self, excl_realized_var: bool):
        """For testing purpose only

        Args:
            excl_realized_var (bool): True - to exclude realized variance from price
        """
        self._excl_realized_var = excl_realized_var

    def get_greeks(self):
        """Return greeks
            A long position of Variance swap has greeks:
                - underlying share: -1
                - replicating options: option_weight
        """
        underlying_price, _ = self._underlying.get_price()
        r = self._underlying.get_risk_free_rate()
        fwd = underlying_price*math.exp(r*self.get_remaining_time())
        hedging_ratios = {self._underlying.get_name(): self._var_notional*1e4*2./self._maturity*(1/fwd-1/self._f)}
        for opt, weight in zip(self._options, self._option_weights):
            hedging_ratios[opt.get_name()] = self._var_notional*1e4*weight*2./self._maturity
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

    def set_pricing_engine(self):
        process_param = self._underlying.get_process_param()
        underlying_price, underlyer_var = self._underlying.get_price()
        assert isinstance(process_param, HestonProcessParam), "Market volatility model has to be dynamic vol (i.e. Heston) for pricing variance swap."
        # Heston model
        self._param = HestonProcessParam(
                risk_free_rate=process_param.risk_free_rate,spot=underlying_price, 
                spot_var=max(consts.IMPLIED_VOL_FLOOR, underlyer_var), 
                drift=process_param.risk_free_rate, dividend=self._underlying.get_dividend_yield(),
                kappa=process_param.kappa, theta=process_param.theta, 
                rho=process_param.rho, vov=process_param.vov, use_risk_free=True
            )

    def get_sim_price(self) -> float:
        """price of simulated episode at current time

        Returns:
            float: price
        """
        cur_time = get_cur_time()
        if (self._cur_price[1] is None):
            # first visit of this timestep, need set up pricing engine
            self.set_pricing_engine()
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
            return self._var_notional*(1e4*self._realized_var - self._var_strike)
        elif (self._maturity - cur_time) <= -1e-5:
            # past expiry
            return 0.
        else:
            # price contains two parts
            # realised variance and future variance expectation
            # future variance
            r = self._underlying.get_risk_free_rate()
            df = math.exp(-r*self.get_remaining_time())
            if self._pricing_method == 'Replicating':
                # Replicating
                fut_variance = self._compute_replication_portfolio(self._option_weights, self._options)
            else:
                # Heston
                fut_variance = self._heston_fut_var()
            if self._excl_realized_var:
                variance = cur_time*self._var_strike/1e4 + self.get_remaining_time()*fut_variance
            else:
                variance = cur_time*self._realized_var + self.get_remaining_time()*fut_variance
            variance = variance/self._maturity
            return df*self._var_notional*(1e4*variance- self._var_strike) 

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
        return self._realized_var*1e4

    def __repr__(self):
        return f'Variance Swap {self._name}: \nunderlying=({str(self._underlying)})\nMaturity={get_period_str_from_time(self._maturity)}, VarStrike={self._var_strike}, VarNotional={self._var_notional}'


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
        'VarSwap AMZN 3M 52 10000 (AMZN_VAR_SWAP)'
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
    

