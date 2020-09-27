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
    __slots__ = ["_ir", "_equity", "_equity_map", "_correlation",
                 "_num_steps", "_time_step", "_implied_vol_surfaces"]

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

    def get_equity(self):
        return self._equity

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
    
    def set_time_step(self, time_step):
        self._time_step = time_step

    def get_time_step(self):
        return self._time_step
    
    def time_step(self, time_step):
        self._time_step = time_step
        return self

    def load_json_data(self, json_: Union[List[dict], str]):
        if isinstance(json_, str):
            eq_list_json = json.loads(json_)
        else:
            eq_list_json = json_
        num_steps = 0; num_paths = 0
        for eq_dict in eq_list_json:
            eq = self._equity_map[eq_dict["name"]]
            eq.load_json_data(eq_dict["data"])
            if num_steps == 0:
                num_steps = eq.get_num_steps()
                num_paths = eq.get_num_paths()
            assert eq.get_num_paths() == num_paths
            assert eq.get_num_steps() == num_steps

    def generate_paths(self, num_paths: float, seed: int=1234):
        """Generate paths
            TODO: add correlation.
        Args:
            num_paths (float): number of paths
            seed (int, optional): RNG seed. Defaults to 1234.
        """
        times = np.linspace(self._time_step, self._num_steps*self._time_step, self._num_steps)
        self._implied_vol_surfaces = dict()
        for eq in self._equity:
            process = eq.get_process()
            paths = process.sample_paths(
                times,
                time_step=self._time_step,
                num_samples=num_paths,
                initial_state=eq.get_initial_state(),
                random_type=random.RandomType.PSEUDO_ANTITHETIC,
                seed=seed
            )
            eq.set_generated_paths(paths, self._time_step)
            eq.set_cur_impvol_path(-1)
            self._implied_vol_surfaces[eq.get_name()] = dict()

    def get_spot(self, equity_name, path_i, step_i=None):
        return self._equity_map[equity_name].get_spot(path_i, step_i)

    def get_implied_vol_surface(self, equity_name, path_i, step_i):
        eq = self._equity_map[equity_name]
        vol_key = f"{path_i}_{step_i}"
        vol_surface_cache = self._implied_vol_surfaces[equity_name]
        if vol_key in vol_surface_cache:
            return vol_surface_cache[vol_key] 
        if eq.get_cur_impvol_path() == np.infty:
            vol_surface_cache[vol_key] = eq.get_impvol(path_i, step_i)
        elif path_i == eq.get_cur_impvol_path():
            vol_surface_cache[vol_key] = eq.get_impvol(path_i, step_i)
        else:
            # generate impvol for path_i
            if eq.get_process_param()["process_type"] == "GBM":
                implied_vols = np.zeros((eq.get_impvol_maturities().shape[0],
                                         eq.get_impvol_strikes().shape[0], self._num_steps),
                                         dtype=np_dtype)
                implied_vols[:] = eq.get_process_param()["param"]["vol"]
                implied_vols = tf.constant(implied_vols, dtype=np_dtype)
            elif eq.get_process_param()["process_type"] == "Heston":
                strikes = eq.get_impvol_strikes()
                maturities = eq.get_impvol_maturities()
                rate = self._ir
                param = eq.get_process_param()["param"]
                mesh_strikes, mesh_maturities, mesh_s0s = tf.meshgrid(strikes,
                                                                      maturities,
                                                                      eq.get_spot(path_i))
                _, _, mesh_vars = tf.meshgrid(strikes,maturities,eq.get_var(path_i))
                dfs = tf.exp(-(rate-param["dividend"])*mesh_maturities)
                mesh_fwds = mesh_s0s/dfs
                prices = tff.models.heston.approximations.european_option_price(
                    variances=mesh_vars,
                    strikes=mesh_strikes,
                    expiries=mesh_maturities,
                    forwards=mesh_fwds,
                    is_call_options=True,
                    kappas=param["kappa"],
                    thetas=param["theta"],
                    sigmas=param["epsilon"],
                    rhos=param["rho"],
                    discount_factors=dfs,
                    dtype=np_dtype)
                initial_volatilities = 0.5
                implied_vols = tff.black_scholes.implied_vol(
                    prices=prices,
                    strikes=mesh_strikes,
                    expiries=mesh_maturities,
                    forwards=mesh_fwds,
                    discount_factors=dfs,
                    is_call_options=True,
                    initial_volatilities=initial_volatilities,
                    validate_args=True,
                    tolerance=1e-9,
                    max_iterations=400,
                    name=None,
                    dtype=np_dtype)  
            eq.set_impvols(implied_vols)
            eq.set_cur_impvol_path(path_i)
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
        return cls().ir(dict_json["ir"]).equity(equities).correlation(correlations)

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
    simulator.set_time_step(1/360)
    simulator.set_num_steps(90)
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
