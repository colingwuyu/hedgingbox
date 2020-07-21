from hb.pricing import blackscholes
from acme import core
from acme import types
import dm_env
from dm_env import specs
import numpy as np


class DeltaHedgeActor(core.Actor):
    """A delta hedge actor,

    An actor based on delta hedge policy which takes market observations
    and outputs hedging actions. 
    """

    def __init__(self,
                 action_spec: specs.BoundedArray):
        self._min_action = action_spec.minimum[0]
        self._max_action = action_spec.maximum[0]

    def select_action(self, observations: types.NestedArray) -> types.NestedArray:
        t = observations[0]
        n_call = observations[2]
        k = observations[3]
        r = observations[4]
        s = observations[6]
        q = observations[8]
        sigma = observations[9]
        cur_holding = observations[10]
        delta = blackscholes.delta(
            call=True, s0=s, r=r, q=q, strike=k, sigma=sigma, tau_e=t, tau_d=t
        )
        action = [np.clip(-delta*n_call - cur_holding, self._min_action, self._max_action)]

        return action

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
