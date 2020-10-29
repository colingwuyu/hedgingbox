from hb.pricing import blackscholes
from hb.instrument.european_option import EuropeanOption
from hb.market_env.portfolio import Portfolio
from acme import core
from acme import types
from dm_env import specs
import dm_env
import numpy as np


class EuroGammaHedgingStrategy:
    """A gamma hedge actor for European Options

    An actor based on delta hedge policy which takes market observations
    and outputs hedging actions. 
    """

    def __init__(self, portfolio: Portfolio,
                 action_spec: specs.BoundedArray):
        self._option_positions = []
        self._hedging_option_positions = {}
        # underlying name => action index mapping
        self._action_index_map = dict()
        # underlying name => holding index in observation mapping
        self._holding_obs_index_map = dict()
        # hedging instruments for european options
        self._hedging_instrument_names = []
        for derivative in portfolio.get_liability_portfolio():
            if isinstance(derivative.get_instrument(), EuropeanOption):
                self._hedging_instrument_names += [derivative.get_instrument().get_underlying_name()]
                self._option_positions += [derivative]
        for position in portfolio.get_hedging_portfolio():
            if isinstance(position.get_instrument(), EuropeanOption):
                self._hedging_option_positions[position.get_instrument().get_underlying_name()] = position
        for i, position in enumerate(portfolio.get_hedging_portfolio()):
            if (position.get_instrument().get_name() in self._hedging_instrument_names) or \
                (position.get_instrument().get_underlying_name() in self._hedging_instrument_names):
                self._action_index_map[position.get_instrument().get_name()] = i
                self._holding_obs_index_map[position.get_instrument().get_name()] = 2*i + 1
            
    def update_action(self, observations: types.NestedArray, actions: types.NestedArray):
        # calculate option's delta, gamma
        delta_map = dict()
        gamma_map = dict()
        for option_position in self._option_positions:
            hedging_name = option_position.get_instrument().get_underlying_name()
            if hedging_name not in delta_map:
                delta_map[hedging_name] = option_position.get_instrument().get_delta() \
                                                                    * option_position.get_holding()  
                gamma_map[hedging_name] = option_position.get_instrument().get_gamma() \
                                                                    * option_position.get_holding()  
            else:
                delta_map[hedging_name] = delta_map[hedging_name] + option_position.get_instrument().get_delta() \
                                                                    * option_position.get_holding()  
                gamma_map[hedging_name] = gamma_map[hedging_name] + option_position.get_instrument().get_gamma() \
                                                                    * option_position.get_holding()  
                
        
        # calculate the buy/sell action from delta        
        for underlying, gamma in gamma_map.items():
            gamma_hedging_option = self._hedging_option_positions[underlying]
            gamma_hedging_name = gamma_hedging_option.get_instrument().get_name()
            hedging_gamma = gamma_hedging_option.get_instrument().get_gamma()
            gamma_hedging_shares = gamma / hedging_gamma if hedging_gamma != 0 else np.infty
            gamma_hedging_cur_holding = self._portfolio.get_position(gamma_hedging_name).get_holding()
            gamma_hedging_action_index = self._action_index_map[gamma_hedging_name]
            action = - gamma_hedging_shares - gamma_hedging_cur_holding
            actions[gamma_hedging_action_index] += action
            gamma_hedging_delta = -(gamma_hedging_cur_holding+action)*gamma_hedging_option.get_instrument().get_delta()
            delta = delta_map[underlying] - gamma_hedging_delta
            underlying_cur_holding = self._portfolio.get_position(underlying).get_holding()
            underlying_action_index = self._action_index_map[underlying]
            action = - delta - underlying_cur_holding
            actions[underlying_action_index] += action
        return actions
