from hb.instrument.instrument import Instrument
from hb.instrument.stock import Stock
from hb.instrument.european_option import EuropeanOption
from hb.instrument.cash_account import CashAccount
from hb.market_env.rewardrules.pnl_reward import PnLReward
from hb.market_env import market_specs
from hb.market_env.rewardrules import reward_rule
from hb.utils.date import *
from hb.utils.process import *
from hb.utils.heston_calibration import *
from hb.market_env.portfolio import Portfolio
import math
from typing import List, Union
import numpy as np
import dm_env
from dm_env import specs



class Market(dm_env.Environment):
    """Market Environment
    """
    def __init__(
            self,
            reward_rule: reward_rule.RewardRule,  
            risk_free_rate: float,
            hedging_step_in_days: int,
        ):
        self._risk_free_rate = risk_free_rate
        self._underlying_processes = dict()
        self._underlying_processes_param = dict()
        self._tradable_products = np.array([])
        self._tradable_products_map = dict()
        self._exotic_products = np.array([])
        self._exotic_products_map = dict()
        self._funding_account = CashAccount(interest_rates=risk_free_rate)
        self._pnl_reward = PnLReward()
        self._reward_rule = reward_rule
        self._hedging_step_in_days = hedging_step_in_days
        self._num_steps = 0
        self._portfolio = None
    
    def add_instrument(self, instrument: Instrument):
        if instrument.get_is_tradable():
            self._tradable_products = np.append(self._tradable_products, instrument)
            self._tradable_products_map[instrument.get_name()] = instrument
        else:
            self._exotic_products = np.append(self._exotic_products, instrument)
            self._exotic_products_map[instrument.get_name()] = instrument

    def add_instruments(self, instruments: List[Instrument]):
        for i in instruments:
            self.add_instrument(i)
 
    def calibrate(self, vol_model: str, 
                  underlying: Stock, 
                  listed_options: Union[EuropeanOption, List[List[EuropeanOption]]]):
        if vol_model == 'BSM':
            param = GBMProcessParam(
                risk_free_rate=self._risk_free_rate,
                spot=underlying.get_quote(), 
                drift=underlying.get_annual_yield(), 
                dividend=underlying.get_dividend_yield(), 
                vol=listed_options.get_quote(),
                use_risk_free=False
            )
        elif vol_model == 'Heston':
            param = heston_calibration(self._risk_free_rate, underlying, listed_options)
        else:
            raise NotImplementedError(f'{vol_model} is not supported')
        underlying.set_pricing_engine(param, self._hedging_step_in_days, self._num_steps)
        self.add_instrument(underlying)
        self.add_instruments(np.array(listed_options).flatten())

    def get_instrument(self, instrument_name: str) -> Instrument:
        if instrument_name in self._tradable_products_map:
            return self._tradable_products_map[instrument_name]
        else:
            return self._exotic_products_map[instrument_name]

    def get_instruments(self, instrument_names: List[str]) -> List[Instrument]:
        return [self.get_instrument(i) for i in instrument_names]

    def init_portfolio(self, portfolio: Portfolio):
        self._num_steps = 0
        for i, h in zip(portfolio.get_instruments(), portfolio.get_holdings()):
            num_steps_to_maturity = int(days_from_time(i.get_maturity_time()) / self._hedging_step_in_days)
            self._num_steps = max(self._num_steps, num_steps_to_maturity)
        self._portfolio = portfolio 
        
    def reset(self):
        reset_date()
        self._portfolio.reset()
        self._funding_account.reset()
        for i in self._portfolio.get_instruments():
            self._set_pricing_engines(i)
        premiums = self._portfolio.get_nav()
        self._funding_account.add(premiums)
        self._reward_rule.reset(self._portfolio)
        self._pnl_reward.reset(self._portfolio)
        return dm_env.restart(np.append(self._observation(), 0.))

    def step(self, action):
        # take action and rebalance hedging
        cashflow, rebalance_cost = self._portfolio.rebalance(action)
        # add any cashflow in/out to funding account
        self._funding_account.add(cashflow)
        # move to next day
        move_days(self._hedging_step_in_days)
        # set pricing model for instruments
        for i in self._portfolio.get_instruments():
            self._set_pricing_engines(i)
        if self._reach_terminal():
            # last step, dump all positions, add cost to transaction cost
            rebalance_cost += self._portfolio.dump_cost()
            ret_step = dm_env.termination(
                reward=self._reward_rule.step_reward(dm_env.StepType.LAST, self._funding_account, 
                                                     self._portfolio, rebalance_cost),
                observation=np.append(self._observation(),
                                      self._pnl_reward.step_reward(dm_env.StepType.LAST, self._funding_account, 
                                                                   self._portfolio, rebalance_cost)))
        else:
            ret_step = dm_env.transition(
                reward=self._pnl_reward.step_reward(dm_env.StepType.MID, self._funding_account, 
                                                    self._portfolio, rebalance_cost),
                observation=np.append(self._observation(),
                                      self._pnl_reward.step_reward(dm_env.StepType.MID, self._funding_account,
                                                                   self._portfolio, rebalance_cost)),
                discount=0.)
        return ret_step
        
    def _reach_terminal(self) -> bool:
        all_expired = True
        for i in self._portfolio.get_instruments():
            if not i.get_is_tradable():
                all_expired = (all_expired and i.get_is_expired())
        return all_expired

    def observation_spec(self):
        obs_shape = (1+2*(len(self._act_exotic_holdings)+len(self._act_tradable_holdings)), )
        return specs.Array(
            shape=obs_shape, dtype=float, name="market_observations"
        )

    def action_spec(self):
        """Returns the action spec.
        """
        maximum = [i.get_trading_limit() for i in self._act_tradable_products]
        minimum = -1*maximum
        discretize_step = [self._market_param.lot_size]
        return specs.BoundedArray(
            shape=self._act_tradable_products.shape, dtype=float,
            minimum=minimum, maximum=maximum, name="hedging_action"
        )

    def _observation(self):
        market_observations = np.zeros(1+2*len(self._portfolio.get_instruments()), dtype=np.float)
        ind = 0
        for i, h in zip(self._portfolio.get_instruments(), 
                        self._portfolio.get_holdings()):
            if isinstance(i, Stock):
                price, _ = i.get_price()
            else:
                price = i.get_price()
            market_observations[ind] = price
            market_observations[ind+1] = h
            ind = ind + 2
        market_observations[ind] = get_cur_time()
        return market_observations
    
    def __repr__(self):
        market_str = 'Market Information: \n'
        market_str += '     Risk Free Rate: \n'
        market_str += f'        {self._risk_free_rate}\n'
        market_str += '     Hedging Step in Days: \n'
        market_str += f'        {self._hedging_step_in_days}\n'
        market_str += '     Reward Rule: \n'
        market_str += f'        {type(self._reward_rule)}\n'
        market_str += f'    Instruments: \n'
        market_str += '         Tradable:\n'
        for i in self._tradable_products:
            market_str += f'            {str(i)}\n'
        market_str += '         Exotic:\n'
        for i in self._exotic_products:
            market_str += f'            {str(i)}\n'
        market_str += f'    Portfolio: \n'
        for i, h in zip(self._portfolio.get_instruments(), self._portfolio.get_holdings()):
            market_str += f'         {i.get_name()} Holding {h}\n'
        return market_str

    
