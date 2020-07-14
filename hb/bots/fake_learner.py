from acme import core
from typing import List
import numpy as np


class FakeLeaner(core.Learner):
    def step(self):
        pass

    def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
        return [[np.array([])]]
