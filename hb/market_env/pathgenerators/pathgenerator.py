import abc
import numpy as np
from acme import types
from typing import Tuple


class PathGenerator(abc.ABC):
    """Interface for stock price path generator.
    """

    def __init__(self, initial_price: float,
                 num_step: int,
                 step_size: float = 1./365.,
                 seed: int = 1234):
        self._initial_price = initial_price
        self._num_step = num_step
        self._step_size = step_size
        self._rng = np.random.RandomState(seed)

    @property
    def episode_steps(self):
        return self._num_step

    @property
    def num_step(self):
        return self._num_step

    @property
    def step_size(self):
        return self._step_size

    @step_size.setter
    def step_size(self, step_size):
        self._step_size = step_size

    @abc.abstractmethod
    def gen_path(self, num_paths: int) -> Tuple[types.NestedArray, types.NestedArray]:
        """Generate number of paths

        return path samples and the attributes feedback to environment 
        which need be included in observation,
        the path includes initial step at time 0
        """

    @abc.abstractmethod
    def gen_step(self, num_steps: int,
                 observation: types.NestedArray, action) -> Tuple[types.NestedArray, types.NestedArray]:
        """Generate number of steps

        return step samples and the attributes feedback to environment 
        which need be included in observation
        """

    @abc.abstractmethod
    def reset_step(self) -> Tuple[types.NestedArray, types.NestedArray]:
        """Reset step 
        """

    def has_env_interaction(self) -> bool:
        """Path Generator Spec 
        Returns True if it needs observation info from environment 
        to produce stock price in next step

        If generator needs interaction with environment for stepping,
        environment calls gen_step instead of gen_path function
        """
        return False
