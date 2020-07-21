from hb.market_env import market_specs
from hb.utils.discretization import int_multiplier
import numpy as np


class QTable:
    def __init__(self,
                 observation_spec: market_specs.DiscretizedBoundedArray,
                 action_spec: market_specs.DiscretizedBoundedArray):
        self._obs_spec = observation_spec
        self._action_spec = action_spec

        mesh = np.meshgrid(*(np.arange(mi, ma+st, st) for mi, ma, st in zip(observation_spec.minimum,
                                                                            observation_spec.maximum,
                                                                            observation_spec.discretize_step)))
        flattened_mesh = np.array([m.flatten() for m in mesh])
        self._qtable = {}
        self._int_multipliers = [int_multiplier(
            disc) for disc in observation_spec.discretize_step]
        for i in range(flattened_mesh.shape[0]):
            flattened_mesh[i] = (flattened_mesh[i] *
                                 self._int_multipliers[i])
        self._flattened_mesh = flattened_mesh.astype(np.int64)
        action_num = round((action_spec.maximum - action_spec.minimum)[0] /
                           action_spec.discretize_step[0]) + 1
        self._action_space = np.arange(action_spec.minimum[0], action_spec.maximum[0]+action_spec.discretize_step[0],
                                       action_spec.discretize_step[0])
        self._qtable = np.zeros((self._flattened_mesh.shape[1], action_num))

    @property
    def qtable(self):
        return self._qtable

    @qtable.setter
    def qtable(self, other_qtable):
        self._qtable = other_qtable

    def copy(self):
        ret_Qtable = QTable(self._obs_spec, self._action_spec)
        ret_Qtable.qtable = self.qtable.copy()
        return ret_Qtable

    @property
    def action_space(self):
        return self._action_space

    def _coding_observation(self, observation: np.ndarray):
        select_ind = np.array([True]*self._flattened_mesh.shape[1])
        for obs_i, obs in enumerate(observation):
            select_ind = select_ind & (self._flattened_mesh[obs_i] == round(obs*self._int_multipliers[obs_i]))
        return np.where(select_ind)[0]

    def select_maxQ_action(self, observation: np.ndarray):
        obs_ind = self._coding_observation(observation)
        argmaxQ_action = self._action_space[np.argmax(self._qtable[obs_ind])]
        return [argmaxQ_action]

    def select_maxQ(self, observation: np.ndarray):
        obs_ind = self._coding_observation(observation)
        return np.max(self._qtable[obs_ind])

    def getQ(self, observation: np.ndarray, action: np.ndarray):
        obs_ind = self._coding_observation(observation)
        action_ind = np.where(self._action_space == action[0])[0]
        return self._qtable[obs_ind][action_ind]

    def update(self, observation: np.ndarray, action: np.ndarray, inc: float):
        obs_hash = hash(observation)
        action_ind = np.where(self._action_space == action[0])[0]
        self._qtable[obs_hash][action_ind] += inc
