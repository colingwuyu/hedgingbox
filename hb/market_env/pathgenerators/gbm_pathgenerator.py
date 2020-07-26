from hb.market_env.pathgenerators import pathgenerator
from acme import types
import numpy as np
from typing import Tuple


class GBMGenerator(pathgenerator.PathGenerator):
    def __init__(self, initial_price: float,
                 drift: float,
                 div: float,
                 sigma: float,
                 num_step: int,
                 step_size: float = 1./360.,
                 seed: int = 1234):
        self._drift = drift
        self._div = div
        self._sigma = sigma
        super().__init__(initial_price, num_step,
                         step_size, seed)

    def gen_path(self, num_paths: int) -> Tuple[types.NestedArray, types.NestedArray]:
        path_samples = np.zeros((num_paths, self._num_step+1))
        path_samples[:, 0] = self._initial_price
        rnds = self._rng.lognormal((self._drift - self._div - self._sigma**2/2) * self._step_size,
                                   self._sigma*np.sqrt(self._step_size),
                                   (num_paths, self._num_step))
        for si in range(1, self._num_step+1):
            path_samples[:, si] = path_samples[:, si-1] * rnds[:, si-1]
        stock_attr = [[{'stock_drift': self._drift,
                        'stock_dividend': self._div,
                        'stock_sigma': self._sigma}]*(self._num_step+1)]*num_paths
        if num_paths == 1:
            path_samples = path_samples[0]
            stock_attr = stock_attr[0]
        return path_samples, stock_attr

    def reset_step(self):
        return self._initial_price, {'stock_drift': self._drift,
                                     'stock_dividend': self._div,
                                     'stock_sigma': self._sigma}

    def gen_step(self, num_step: int,
                 observation: types.NestedArray, action) -> Tuple[types.NestedArray, types.NestedArray]:
        step_samples = np.zeros(num_step + 1)
        step_samples[0] = observation['stock_price']
        rnds =  self._rng.lognormal((self._drift - self._div - self._sigma**2/2) * self._step_size,
                                    self._sigma*np.sqrt(self._step_size),
                                    self._num_step)
        for si in range(1, num_step+1):
            step_samples[si] += step_samples[si-1] * rnds[si-1]
        step_samples = step_samples[1:]
        stock_attr = [{'stock_drift': self._drift,
                       'stock_dividend': self._div,
                       'stock_sigma': self._sigma}]*num_step
        if num_step == 1:
            stock_attr = stock_attr[0]
        return step_samples, stock_attr
