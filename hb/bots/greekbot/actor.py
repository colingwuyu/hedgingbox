from hb.pricing import blackscholes
from hb.instrument.european_option import EuropeanOption
from hb.market_env.portfolio import Portfolio
from hb.bots.greekbot.hedging_strategy import *
from acme import core
from acme import types
from dm_env import specs
import dm_env
import numpy as np
from hb.utils.consts import np_dtype


class GreekHedgeActor(core.Actor):
    """A delta hedge actor for European Options

    An actor based on delta hedge policy which takes market observations
    and outputs hedging actions. 
    """

    def __init__(self, portfolio: Portfolio,
                 action_spec: specs.BoundedArray,
                 strategies = [EuroDeltaHedgingStrategy, VarianceSwapReplicatingStrategy]
                 ):
        self._strategies = [
            strategy(portfolio, action_spec) for strategy in strategies
        ]
        self._actions = np.zeros(action_spec.shape)
        self._hedging_positions = portfolio.get_hedging_portfolio()

    def select_action(self, observations: types.NestedArray) -> types.NestedArray:
        actions = self._actions.copy()
        for strategy in self._strategies:
            actions = strategy.update_action(observations, actions)
        for action_i, position in enumerate(self._hedging_positions):
            limit = np_dtype(position.get_instrument().get_trading_limit())
            actions[action_i] = np.max(-limit, np.min(actions[action_i], limit))/limit
        return actions

    def observe_first(self, timestep: dm_env.TimeStep):
        pass

    def observe(
        self,
        action: types.NestedArray,
        next_timestep: dm_env.TimeStep
    ):
        pass

    def update(self):
        pass
