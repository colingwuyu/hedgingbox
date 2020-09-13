from hb.pricing import blackscholes
from hb.instrument.variance_swap import VarianceSwap
from hb.market_env.portfolio import Portfolio
from acme import core
from acme import types
from dm_env import specs
import dm_env
import numpy as np


class VarianceSwapReplicatingStrategy:
    """A replicating hedge actor for variance swaps

    An actor based on replicating portfolio policy which takes market observations
    and outputs hedging actions. 
    """

    def __init__(self, portfolio: Portfolio,
                 action_spec: specs.BoundedArray):
        self._use_bs_delta = use_bs_delta
        self._min_action = action_spec.minimum
        self._max_action = action_spec.maximum
        # underlying name => action index mapping
        self._action_index_map = dict()
        # underlying name => holding index in observation mapping
        self._holding_obs_index_map = dict()
        # hedging instruments for european options
        self._hedging_ratios = {}
        # TODO add up hedging instruments and hedging ratios
        for derivative in portfolio.get_liability_portfolio():
            if isinstance(derivative.get_instrument(), VarianceSwap):
                for hedging_name, hedging_ratio in derivative.get_instrument().get_greeks().items():
                    if hedging_name in self._hedging_ratios:
                        self._hedging_ratios[hedging_name] += hedging_ratio*derivative.get_holding()
                    else:
                        self._hedging_ratios[hedging_name] = hedging_ratio*derivative.get_holding()
        for i, position in enumerate(portfolio.get_hedging_portfolio()):
            if position.get_instrument().get_name() in self._hedging_ratios:
                self._action_index_map[position.get_instrument().get_name()] = i
                self._holding_obs_index_map[position.get_instrument().get_name()] = 2*i + 1

    def update_action(self, observations: types.NestedArray, actions: types.NestedArray):
        # calculate the buy/sell action from delta        
        for underlying, delta in self._hedging_ratios.items():
            holding_obs_index = self._holding_obs_index_map[underlying]
            cur_holding = observations[holding_obs_index]
            action_index = self._action_index_map[underlying]
            action = np.clip(- delta - cur_holding, self._min_action[action_index], self._max_action[action_index])
            actions[action_index] += action
        return actions
