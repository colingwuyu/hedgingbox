from hb.pricing import blackscholes
from hb.market_env import market_specs
from acme import core
from acme import types
import dm_env
import numpy as np


class NoHedgeActor(core.Actor):
    """A no hedge actor,

    An actor alsways takes 0 action. 
    """

    def __init__(self,
                 action_spec: market_specs.DiscretizedBoundedArray):
        self._action = np.zeros(action_spec.shape)

    def select_action(self, observations: types.NestedArray) -> types.NestedArray:
        return self._action

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

    def obs_attr_requirement(self):
        return []
