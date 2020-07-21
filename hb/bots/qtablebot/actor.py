from acme import adders
from acme import core
from acme import types

import dm_env
import numpy as np

from hb.bots.qtablebot import qtable


class QTableActor(core.Actor):
    """A feed-forward actor.

    An actor based on a feed-forward policy which takes non-batched observations
    and outputs non-batched actions. It also allows adding experiences to replay
    and updating the weights from the policy on the learner.
    """

    def __init__(
        self,
        qtable: qtable.QTable,
        epsilon: float,
        adder: adders.Adder = None
    ):
        """Initializes the actor.

        Args:
          qtable: the QTable policy to run.
          epsilon: epsilon of greedy method for exploration
          adder: the adder object to which allows to add experiences to a
            dataset/replay buffer.
        """

        # Store these for later use.
        self._adder = adder
        self._epsilon = epsilon
        self._qtable = qtable

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        rnd_tmp = np.random.random(1)
        if rnd_tmp < self._epsilon:
            return np.random.choice(self._qtable.action_space, 1).astype(np.float32)
        else:
            return self._qtable.select_maxQ_action(observation)

    def observe_first(self, timestep: dm_env.TimeStep):
        if self._adder:
            self._adder.add_first(timestep)

    def observe(
        self,
        action: types.NestedArray,
        next_timestep: dm_env.TimeStep,
    ):
        if self._adder:
            self._adder.add(action, next_timestep)

    @property
    def exploration_epsilon(self):
        return self._epsilon

    @exploration_epsilon.setter
    def exploration_epsilon(self, epsilon):
        self._epsilon = epsilon

    def update(self):
        pass
