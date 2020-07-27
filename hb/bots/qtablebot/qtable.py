from hb.market_env import market_specs
from hb.utils.discretization import int_multiplier
import numpy as np
import secrets


class QTable:
    def __init__(self,
                 observation_spec: market_specs.DiscretizedBoundedArray,
                 action_spec: market_specs.DiscretizedBoundedArray):
        self._obs_spec = observation_spec
        self._action_spec = action_spec
        self._int_multipliers = [int_multiplier(
            disc) for disc in observation_spec.discretize_step]
        action_num = round((action_spec.maximum - action_spec.minimum)[0] /
                           action_spec.discretize_step[0]) + 1
        self._action_space = np.arange(action_spec.minimum[0], action_spec.maximum[0]+action_spec.discretize_step[0],
                                       action_spec.discretize_step[0])
        self._action_init = np.zeros(int(action_num))
        self.qtable = {}

    def _encoding_obs(self, observation: np.ndarray):
        encoded_obs = observation.copy().astype(str)
        for obs_i, obs in enumerate(observation):
            encoded_obs[obs_i] = str(int(round(obs * self._int_multipliers[obs_i])))
        return "|".join(encoded_obs)

    @property
    def qtable(self):
        return self._qtable

    @qtable.setter
    def qtable(self, other_qtable):
        self._qtable = other_qtable

    def copy(self):
        ret_Qtable = QTable(self._obs_spec, self._action_spec)
        for k, v in self._qtable.items():
            ret_Qtable.qtable[k] = v.copy()
        return ret_Qtable

    @property
    def action_space(self):
        return self._action_space

    def select_maxQ_action(self, observation: np.ndarray):
        obs_ind = self._encoding_obs(observation)
        if obs_ind not in self._qtable:
            self._qtable[obs_ind] = self._action_init.copy()
        action_arr = self._qtable[obs_ind]
        action_maxQ_ind = np.where(action_arr==np.max(action_arr))
        action_ind = secrets.choice(action_maxQ_ind[0])
        argmaxQ_action = self._action_space[action_ind]
        return np.array([argmaxQ_action.astype(np.float32)])

    def select_maxQ(self, observation: np.ndarray):
        obs_ind = self._encoding_obs(observation)
        if obs_ind not in self._qtable:
            self._qtable[obs_ind] = self._action_init.copy()
        return np.max(self._qtable[obs_ind])

    def getQ(self, observation: np.ndarray, action: np.ndarray):
        obs_ind = self._encoding_obs(observation)
        if obs_ind not in self._qtable:
            self._qtable[obs_ind] = self._action_init.copy()
        action_ind = np.where(self._action_space == action[0])[0]
        return self._qtable[obs_ind][action_ind][0]

    def update(self, observation: np.ndarray, action: np.ndarray, inc: float):
        obs_ind = self._encoding_obs(observation)
        if obs_ind not in self._qtable:
            self._qtable[obs_ind] = self._action_init.copy()
        action_ind = np.where(self._action_space == action[0])[0]
        self._qtable[obs_ind][action_ind] += inc
