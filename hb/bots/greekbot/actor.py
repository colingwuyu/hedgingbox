from hb.pricing import blackscholes
from hb.instrument.european_option import EuropeanOption
from hb.market_env.portfolio import Portfolio
from hb.bots.greekbot.hedging_strategy import EuroDeltaHedgingStrategy
from acme import core
from acme import types
from dm_env import specs
import dm_env
import numpy as np


class GreekHedgeActor(core.Actor):
    """A delta hedge actor for European Options

    An actor based on delta hedge policy which takes market observations
    and outputs hedging actions. 
    """

    def __init__(self, portfolio: Portfolio, use_bs_delta: bool,
                 action_spec: specs.BoundedArray):
        self._strategies = [
            EuroDeltaHedgingStrategy(portfolio, use_bs_delta, action_spec)
        ]
        self._actions = np.zeros(action_spec.shape)

    def select_action(self, observations: types.NestedArray) -> types.NestedArray:
        actions = self._actions.copy()
        for strategy in self._strategies:
            actions = strategy.update_action(observations, actions)
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
