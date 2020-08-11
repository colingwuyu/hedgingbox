from acme.agents import agent
from acme import core
from acme import types
from hb.core import Predictor
from hb.core import ActorAdapter
import dm_env


class Bot(agent.Agent):
    """Bot class which derives agent.Agent

    Include predition into environment loop and perogress logger.
    In addition to actor and learner in agent.Agent, it includes a predictor which
    assesses the progress of learning performance.

    """

    def __init__(self, actor: core.Actor, learner: core.Learner, predictor: Predictor,
                 min_observations: int, observations_per_step: float,
                 pred_episods: int, observations_per_pred: int,
                 pred_only: bool = False):
        self._predictor = predictor
        self._observations_per_pred = observations_per_pred
        self._pred_episods = pred_episods
        self._cur_episods = -self._observations_per_pred
        self._pred = False
        self._pred_only = pred_only
        self.set_pred_only(pred_only)
        super().__init__(ActorAdapter(actor), learner,
                         min_observations, observations_per_step)

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        if self._pred:
            return self._predictor.select_action(observation)
        else:
            return self._actor.select_action(observation)

    def observe_first(self, timestep: dm_env.TimeStep):
        if self._pred:
            self._predictor.observe_first(timestep)
        else:
            self._actor.observe_first(timestep)

    def observe(
        self,
        action: types.NestedArray,
        next_timestep: dm_env.TimeStep,
    ):
        if self._pred:
            self._predictor.observe(action, next_timestep)
        else:
            super().observe(action, next_timestep)
        if next_timestep.last():
            self._cur_episods += 1
            if self._cur_episods == 0:
                if self._pred and (not self._pred_only):
                    self._predictor.log_progress()
                if self._pred_only:
                    self._cur_episods = -self._pred_episods
                else:
                    self._pred = not self._pred
                    self._cur_episods = -self._pred_episods if self._pred else -self._observations_per_pred

    def update(self):
        if not (self._pred or self._pred_only):
            super().update()
            if (self._num_observations >= 0 and
                    self._num_observations % self._observations_per_update == 0):
                self._predictor.update()

    def set_pred_only(self, pred_only: bool):
        self._pred_only = pred_only
        if self._pred_only:
            self._pred = True
            self._cur_episods = -self._pred_episods
            self._predictor.start_log_perf()
        else:
            self._pred = False
            self._cur_episods = -self._observations_per_pred
            self._predictor.end_log_perf()
