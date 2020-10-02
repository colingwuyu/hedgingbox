import abc
from acme import types
import dm_env
from typing import Dict


class RewardRule(abc.ABC):
    """Interface for reward rule.
    """

    @abc.abstractmethod
    def step_reward(self, step_type: dm_env.StepType,
                    next_step_obs: Dict, 
                    action: types.NestedArray,
                    extra: dict) -> types.NestedArray:
        """Generate reward for the step
        """

    @abc.abstractmethod
    def reset(self, reset_observation):
        """Reset the buffers for new episode
        """
