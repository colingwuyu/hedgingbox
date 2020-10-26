from typing import List, Union
import json
import numpy as np
import pandas as pd
from hb.utils.consts import np_dtype
from hb.utils.string import *
from hb.riskfactorsimulator.equity import Equity
import tf_quant_finance as tff
import tensorflow as tf
from tf_quant_finance.math import random_ops as random


class Correlation(object):
    __slots__ = ["_equity1", "_equity2", "_corr"]
    
    def set_equity1(self, equity1: str):
        self._equity1 = equity1

    def get_equity1(self) -> str:
        return self._equity1

    def equity1(self, equity1):
        self._equity1 = equity1
        return self

    def set_equity2(self, equity2: str):
        self._equity2 = equity2

    def get_equity2(self) -> str:
        return self._equity2

    def equity2(self, equity2):
        self._equity2 = equity2
        return self
    
    def set_corr(self, corr):
        self._corr = corr
    
    def get_corr(self):
        return self._corr
    
    def corr(self, corr):
        self._corr = corr
        return self

    @classmethod
    def load_json(cls, json_: Union[str, dict]):
        if isinstance(json_, str):
            dict_json = json.loads(json_)
        else:
            dict_json = json_
        return cls().equity1(dict_json["equity1"]).equity2(dict_json["equity2"]).corr(dict_json["corr"])

    def jsonify_dict(self) -> dict:
        return {"equity1": self._equity1, "equity2": self._equity2, "corr": self._corr}
    
    def __repr__(self):
        return json.dumps(self.jsonify_dict(), indent=4)

class Simulator(object):
    __slots__ = ["_ir", "_equity", "_equity_map", "_correlation", "_num_paths",
                 "_num_steps", "_time_step", "_implied_vol_surfaces",
                 "_rng_seed"]

    def set_ir(self, ir: float):
        self._ir = ir

    def get_ir(self):
        return self._ir

    def ir(self, ir):
        self._ir = ir
        return self

    def set_equity(self, equity: List[Equity]):
        self._equity = equity
        self._equity_map = {eq.get_name(): eq for eq in equity}

    def get_equity(self, equity_name):
        return self._equity_map[equity_name]

    def equity(self, equity):
        self.set_equity(equity)
        return self

    def set_correlation(self, correlation: List[Correlation]):
        self._correlation = correlation

    def get_correlation(self):
        return self._correlation

    def correlation(self, correlation: List[Correlation]):
        self._correlation = correlation
        return self

    def set_num_steps(self, num_steps):
        self._num_steps = num_steps
    
    def get_num_steps(self):
        return self._num_steps

    def num_steps(self, num_steps):
        self._num_steps = num_steps
        return self

    def set_num_paths(self, num_paths):
        self._num_paths = num_paths
    
    def get_num_paths(self):
        return self._num_paths

    def num_paths(self, num_paths):
        self._num_paths = num_paths
        return self
    
    def set_time_step(self, time_step):
        self._time_step = time_step

    def get_time_step(self):
        return self._time_step
    
    def time_step(self, time_step):
        self._time_step = time_step
        return self

    def set_rng_seed(self, rng_seed):
        self._rng_seed = rng_seed
    
    def get_rng_seed(self):
        return self._rng_seed
    
    def rng_seed(self, rng_seed):
        self._rng_seed = rng_seed
        return self

    def load_json_data(self, json_: Union[List[dict], str]):
        if isinstance(json_, str):
            eq_list_json = json.loads(json_)
        else:
            eq_list_json = json_
        num_steps = 0; num_paths = 0; time_step = 0
        self._implied_vol_surfaces = dict()
        for eq_dict in eq_list_json["scenario"]:
            eq = self._equity_map[eq_dict["name"]]
            eq.load_json_data(eq_dict["data"])
            if num_steps == 0:
                num_steps = eq.get_num_steps()
                num_paths = eq.get_num_paths()
                time_step = eq.get_time_step()
            assert eq.get_num_paths() == num_paths
            assert eq.get_num_steps() == num_steps
            assert eq.get_time_step() == time_step
            self._implied_vol_surfaces[eq.get_name()] = dict()
        self._num_paths = num_paths
        # num steps exclude initial state at time 0
        self._num_steps = num_steps - 1
        self._time_step = time_step

    def _corr_structure(self):
        eq_names = [eq.get_name() for eq in self._equity]
        corr_structure = pd.DataFrame(np.ones([len(self._equity),len(self._equity)]), 
                                     index=eq_names, columns=eq_names)
        for corr in self._correlation:
            corr_structure.loc[corr.get_equity1(),corr.get_equity2()] = corr.get_corr()
            corr_structure.loc[corr.get_equity2(),corr.get_equity1()] = corr.get_corr()
        extra_corr = 0
        for eq in self._equity:
            if eq.get_process_param()["process_type"] == "Heston":
                extra_corr += 1
        extra_corr_structure = [[1.] for i in range(extra_corr)]
        if len(extra_corr_structure) > 0:
            corr_structure = [corr_structure.values.tolist(), extra_corr_structure]
        else:
            corr_structure = [corr_structure.values.tolist()]
        return corr_structure

    def generate_paths(self, num_paths: float, seed: int=None):
        """Generate paths
            TODO: correlation now only supports GBM.
        Args:
            num_paths (float): number of paths
            seed (int, optional): RNG seed. Defaults to None.
        """
        if seed is None:
            seed = self._rng_seed
        times = np.linspace(self._time_step, self._num_steps*self._time_step, self._num_steps)
        if hasattr(self, "_implied_vol_surfaces"):
            if self._implied_vol_surfaces is None:
                self._implied_vol_surfaces = dict()
        else:
            self._implied_vol_surfaces = dict()

        processes = []
        initial_state = []
        for eq in self._equity:
            processes.append(eq.get_process())
            initial_state.append(eq.get_initial_state())
        join_process = tff.models.JoinedItoProcess(
            processes=processes, corr_structure=self._corr_structure()
        )
        all_paths = join_process.sample_paths(
            times,
            time_step=self._time_step,
            num_samples=num_paths,
            initial_state=np.array(initial_state).T,
            random_type=random.RandomType.PSEUDO_ANTITHETIC,
            seed=seed
        )

        cur_path_j = 0
        for i, eq in enumerate(self._equity):
            if eq.get_process_param()["process_type"] == "GBM":
                paths = all_paths[:,:,cur_path_j:(cur_path_j+1)]
                cur_path_j += 1
            elif eq.get_process_param()["process_type"] == "Heston":
                paths = all_paths[:,:,cur_path_j:(cur_path_j+2)]
                cur_path_j += 2
            if hasattr(eq, "_spots"):
                if eq.get_spots() is None:
                    eq.set_generated_paths(paths, self._time_step, self._ir)
                    self._implied_vol_surfaces[eq.get_name()] = dict()
            else:
                eq.set_generated_paths(paths, self._time_step, self._ir)
                self._implied_vol_surfaces[eq.get_name()] = dict()
           
    def get_spot(self, equity_name, path_i, step_i=None):
        return self._equity_map[equity_name].get_spot(path_i, step_i)

    def get_implied_vol_surface(self, equity_name, path_i, step_i):
        eq = self._equity_map[equity_name]
        vol_key = f"{path_i}_{step_i}"
        vol_surface_cache = self._implied_vol_surfaces[equity_name]
        if vol_key in vol_surface_cache:
            return vol_surface_cache[vol_key] 
        vol_surface_cache[vol_key] = eq.get_impvol(path_i, step_i)
        return vol_surface_cache[vol_key] 

    @classmethod
    def load_json(cls, json_: Union[dict, str]):
        if isinstance(json_, str):
            dict_json = json.loads(json_)
        else:
            dict_json = json_
        equities = []
        for eq in dict_json["equity"]:
            equities += [Equity.load_json(eq)]
        correlations = []
        for corr in dict_json["correlation"]:
            correlations += [Correlation.load_json(corr)]
        return cls().ir(dict_json["ir"]).equity(equities).correlation(correlations).rng_seed(4321)

    def jsonify_dict(self) -> dict:
        dict_json = dict()
        dict_json["ir"] = self._ir
        dict_json["equity"] = [eq.jsonify_dict() for eq in self._equity]
        dict_json["correlation"] = [corr.jsonify_dict() for corr in self._correlation]
        return dict_json

    def __repr__(self):
        return json.dumps(self.jsonify_dict(), indent=4)


if __name__ == "__main__":
    test_json ="""
    {
        "ir": 0.015,
        "equity": [
            {
                "name": "AMZN",
                "riskfactors": ["Spot", 
                                "Vol 3Mx100",
                                "Vol 2Mx100",
                                "Vol 4Wx100"],
                "process_param": {
                    "process_type": "Heston",
                    "param": {
                        "spot": 100,
                        "spot_var": 0.096024,
                        "drift": 0.25,
                        "dividend": 0.0,
                        "kappa": 6.288453,
                        "theta": 0.397888,
                        "epsilon": 0.753137,
                        "rho": -0.696611
                    } 
                }
            },
            {
                "name": "SPX",
                "riskfactors": ["Spot", "Vol 3Mx100"],
                "process_param": {
                    "process_type": "GBM",
                    "param": {
                        "spot": 100,
                        "drift": 0.10,
                        "dividend": 0.01933,
                        "vol": 0.25
                    } 
                }
            }
        ],
        "correlation": [
            {
                "equity1": "AMZN",
                "equity2": "SPX",
                "corr": 0.8
            }
        ]
    }
    """
    simulator = Simulator.load_json(test_json)
    with open("sim.json", 'w') as sim_file:
        sim_file.write(str(simulator))
    test_data = json.loads("""
    [
        {
            "name": "AMZN",
            "data": {
                        "time_step_day": 1,
                        "Spot": [3400.0,3414.063507540342,3360.1097430892696,3514.713081433771,3399.4403346846934,3388.775188349936,3296.0554086124134,3330.74487143777],
                        "Vol 3Mx100": [0.3321,0.3321,0.3321,0.3321,0.3321,0.3321,0.3321,0.3321]
                    }
        }
    ]
    """)
    simulator.load_json_data(test_data)
    print(simulator)
    simulator.set_time_step(1/360)
    simulator.set_num_steps(90)
    simulator.generate_paths(100)
    print(simulator.get_spot("AMZN",path_i=0,step_i=1))
    imp_vol_surf = simulator.get_implied_vol_surface("AMZN",path_i=0,step_i=30)
    print(imp_vol_surf.get_black_vol(t=60/360,k=100.))
    # import matplotlib.pyplot as plt
    # for i in range(100):
    #     plt.plot(simulator.get_spot("AMZN", path_i=i))
    # plt.show()
    for path_i in range(100):
        for step_i in range(90):
            print(simulator.get_implied_vol_surface("AMZN", path_i=path_i, step_i=step_i).get_black_vol(t=90/360-step_i/360,k=100.))
    print(simulator.get_spot("SPX",path_i=0,step_i=1))
    imp_vol_surf = simulator.get_implied_vol_surface("SPX",path_i=0,step_i=30)
    print(imp_vol_surf.get_black_vol(t=60/360,k=100.))
