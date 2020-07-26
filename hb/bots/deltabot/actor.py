from hb.pricing import blackscholes
from hb.market_env import market_specs
from acme import core
from acme import types
import dm_env
import numpy as np


class DeltaHedgeActor(core.Actor):
    """A delta hedge actor,

    An actor based on delta hedge policy which takes market observations
    and outputs hedging actions. 
    """

    def __init__(self,
                 action_spec: market_specs.DiscretizedBoundedArray):
        self._min_action = action_spec.minimum[0]
        self._max_action = action_spec.maximum[0]
        self._action_lot = action_spec.discretize_step[0]

    def select_action(self, observations: types.NestedArray) -> types.NestedArray:
        t = observations[0]/360.
        n_call = observations[1]
        k = observations[2]
        r = observations[3]
        s = observations[4]
        q = observations[5]
        sigma = observations[6]
        cur_holding = observations[7]
        delta = blackscholes.delta(
            call=True, s0=s, r=r, q=q, strike=k, sigma=sigma, tau_e=t, tau_d=t
        )
        action = np.clip(-delta*n_call - cur_holding, self._min_action, self._max_action)
        if self._action_lot != 0.:
            action = [round(action/self._action_lot)*self._action_lot]
        else:
            action = [action]
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

    def obs_attr_requirement(self):
        return ['remaining_time', 'option_holding', 'option_strike',
                'interest_rate', 'stock_price', 'stock_dividend',
                'stock_sigma', 'stock_holding']
