from hb.pricing import blackscholes
from hb.instrument.european_option import EuropeanOption
from hb.market_env.portfolio import Portfolio
from acme import core
from acme import types
from dm_env import specs
import dm_env
import numpy as np


class EuroDeltaHedgingStrategy:
    """A delta hedge actor for European Options

    An actor based on delta hedge policy which takes market observations
    and outputs hedging actions. 
    """

    def __init__(self, portfolio: Portfolio,
                 action_spec: specs.BoundedArray):
        self._min_action = action_spec.minimum
        self._max_action = action_spec.maximum
        self._option_positions = []
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
        for i, position in enumerate(portfolio.get_hedging_portfolio()):
            if position.get_instrument().get_name() in self._hedging_instrument_names:
                self._action_index_map[position.get_instrument().get_name()] = i
                self._holding_obs_index_map[position.get_instrument().get_name()] = 2*i + 1

    def update_action(self, observations: types.NestedArray, actions: types.NestedArray):
        delta_map = {k: 0. for k, _ in self._action_index_map.items()}
        
        # calculate option's delta
        for option_position in self._option_positions:
            hedging_name = option_position.get_instrument().get_underlying_name()
            delta_map[hedging_name] = delta_map[hedging_name] + option_position.get_instrument().get_delta() \
                                                                * option_position.get_holding()  
        
        # calculate the buy/sell action from delta        
        for underlying, delta in delta_map.items():
            holding_obs_index = self._holding_obs_index_map[underlying]
            cur_holding = observations[holding_obs_index]
            action_index = self._action_index_map[underlying]
            action = np.clip(- delta - cur_holding, self._min_action[action_index], self._max_action[action_index])
            actions[action_index] += action
        return actions
