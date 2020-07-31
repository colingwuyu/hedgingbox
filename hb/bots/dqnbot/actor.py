from acme import adders
from acme import types
from acme.tf import variable_utils as tf2_variable_utils
from acme.agents.tf import actors

import sonnet as snt
import dm_env

import numpy as np
from hb.market_env import market_specs


class DQNActor(actors.FeedForwardActor):
    """A hedging actor.

    It selects the buy-sell actions from network policy
    """

    def __init__(
        self,
        policy_network: snt,
        action_spec: market_specs.DiscretizedBoundedArray,
        adder: adders.Adder = None,
        variable_client: tf2_variable_utils.VariableClient = None,
    ):
        self._action_space = np.arange(action_spec.minimum[0], action_spec.maximum[0]+action_spec.discretize_step[0],
                                       action_spec.discretize_step[0])
        super().__init__(policy_network, adder, variable_client)

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        action_ind = super().select_action(observation)
        return np.array([self._action_space[action_ind].astype(np.float32)])

    def observe(self,
                action: types.NestedArray,
                next_timestep: dm_env.TimeStep,
                ):
        action = action[0]
        super().observe(action, next_timestep)
