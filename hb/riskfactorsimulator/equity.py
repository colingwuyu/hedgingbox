from typing import List, Union
import json
import numpy as np
import tf_quant_finance as tff
import tensorflow as tf
from hb.utils.consts import *
from tf_quant_finance.math import random_ops as random
from hb.utils.string import *
from hb.utils.date import *
from hb.riskfactorsimulator.impliedvolsurface import ImpliedVolSurface

def parse_impvol_name(impvol_name):
    str_maturity=impvol_name[4:].split("x")[0]
    str_strike=impvol_name[4:].split("x")[1]
    return get_period_from_str(str_maturity), float(str_strike)

class Equity(object):
    __slots__ = ["_name", "_riskfactors", "_process_param", 
                 "_impvol_maturities", "_impvol_strikes",
                 "_str_impvol_maturities", "_str_impvol_strikes",
                 "_impvols", "_spots", "_vars", "_num_paths", "_num_steps",
                 "_time_step"]

    def set_name(self, name):
        self._name = name
    
    def get_name(self):
        return self._name
    
    def name(self, name):
        self._name = name
        return self

    def set_riskfactors(self, riskfactors: List[str]):
        self._riskfactors = riskfactors
        _maturities = set()
        _strikes = set()
        for rf in riskfactors:
            if "Vol" in rf:
                m, k = parse_impvol_name(rf)
                _maturities.add(m)
                _strikes.add(k)
        assert len(riskfactors)-1 == len(_maturities)*len(_strikes), "Need a surface for implied vol risk factors" 
        self._str_impvol_maturities = np.sort(np.array([float_to_str(im) for im in _maturities]))
        self._str_impvol_strikes = np.sort(np.array([float_to_str(ik) for ik in _strikes]))
        self._impvol_maturities = tf.constant(np.sort(np.array([im for im in _maturities], dtype=np_dtype)))
        self._impvol_strikes = tf.constant(np.sort(np.array([ik for ik in _strikes], dtype=np_dtype)))

    def get_riskfactors(self):
        return self._riskfactors

    def riskfactors(self, riskfactors):
        self.set_riskfactors(riskfactors)
        return self

    def set_process_param(self, process_param):
        self._process_param = process_param

    def get_process_param(self):
        return self._process_param
    
    def process_param(self, process_param):
        self._process_param = process_param
        return self

    def get_process(self):
        param = self._process_param["param"]
        drift = tf.constant([param["drift"]], dtype=np_dtype)
        div = tf.constant([param["dividend"]], dtype=np_dtype)
        if self._process_param["process_type"] == "GBM":
            sigma = tf.constant([param["vol"]], dtype=np_dtype)
            def drift_fn(t, x):
                del t, x
                return drift - div - 0.5 * sigma**2
            def vol_fn(t, x):
                del t, x
                return tf.reshape(sigma, [1, 1])
            return tff.models.GenericItoProcess(
                        dim=1,
                        drift_fn=drift_fn,
                        volatility_fn=vol_fn,
                        dtype=np_dtype
                    )
        elif self._process_param["process_type"] == "Heston":
            return tff.models.HestonModel(kappa=param["kappa"],
                                          theta=param["theta"], 
                                          epsilon=param["epsilon"], 
                                          rho=param["rho"],
                                          dtype=np_dtype)

    def get_initial_state(self):
        param = self._process_param["param"]
        if self._process_param["process_type"] == "GBM":
            return np.array([np.log(param["spot"])])
        elif self._process_param["process_type"] == "Heston":
            return np.array([np.log(param["spot"]), param["spot_var"]])

    def set_generated_paths(self, paths, time_step, rate=None):
        param = self._process_param["param"]
        self._num_paths = paths.shape[0]
        self._num_steps = paths.shape[1]
        self._time_step = time_step
        if self._process_param["process_type"] == "GBM":
            self._spots = tf.math.exp(paths[:,:,0])
            implied_vols = np.zeros((self._impvol_maturities.shape[0],
                                     self._impvol_strikes.shape[0]),
                                    dtype=np_dtype)
            implied_vols[:] = param["vol"]
            implied_vols = tf.constant(implied_vols, dtype=np_dtype)
        elif self._process_param["process_type"] == "Heston":
            stocks_no_drift = tf.math.exp(paths[:,:,0])
            times = np.linspace(self._time_step, self._num_steps*self._time_step, self._num_steps, dtype=np_dtype)
            self._vars = paths[:,:,1]
            self._spots = stocks_no_drift*tf.exp((param["drift"]-param["dividend"])*times)
            self._vars = tf.concat([np.array([[param["spot_var"]]]*self._vars.shape[0]),self._vars],-1)
            mesh_strikes, mesh_maturities = tf.meshgrid(self._impvol_strikes,self._impvol_maturities)
            initial_volatilities = 0.5
            implied_vols = np.zeros(mesh_strikes.shape+self._spots.shape)
            for mesh_maturity_i in range(mesh_strikes.shape[0]):
                for mesh_strike_i in range(mesh_strikes.shape[1]):
                    dfs = tf.exp(-(rate-param["dividend"])*mesh_maturities[mesh_maturity_i][mesh_strike_i])
                    fwds = self._spots/dfs
                    prices = tff.models.heston.approximations.european_option_price(
                        variances=self._vars,
                        strikes=mesh_strikes[mesh_maturity_i][mesh_strike_i],
                        expiries=mesh_maturities[mesh_maturity_i][mesh_strike_i],
                        forwards=fwds,
                        is_call_options=True,
                        kappas=param["kappa"],
                        thetas=param["theta"],
                        sigmas=param["epsilon"],
                        rhos=param["rho"],
                        discount_factors=dfs,
                        dtype=np_dtype)
                    implied_vol_slice = tff.black_scholes.implied_vol(
                        prices=prices,
                        strikes=mesh_strikes[mesh_maturity_i][mesh_strike_i],
                        expiries=mesh_maturities[mesh_maturity_i][mesh_strike_i],
                        forwards=fwds,
                        discount_factors=dfs,
                        is_call_options=True,
                        initial_volatilities=initial_volatilities,
                        validate_args=True,
                        tolerance=1e-9,
                        max_iterations=400,
                        name=None,
                        dtype=None)
                    implied_vols[mesh_maturity_i, mesh_strike_i,:,:] = implied_vol_slice
        self._impvols = implied_vols
        self._spots = tf.concat([np.array([[param["spot"]]]*self._spots.shape[0]),self._spots],-1)

    def get_impvol_maturities(self):
        return self._impvol_maturities

    def set_impvol_maturities(self, impvol_maturities):
        self._impvol_maturities = impvol_maturities

    def impvol_maturities(self, impvol_maturities):
        self._impvol_maturities = impvol_maturities
        return self

    def get_impvol_strikes(self):
        return self._impvol_strikes

    def set_impvol_strikes(self, impvol_strikes):
        self._impvol_strikes = impvol_strikes
    
    def impvol_strikes(self, impvol_strikes):
        self._impvol_strikes = impvol_strikes
        return self

    def get_spots(self):
        return self._spots
    
    def set_spots(self, spots):
        self._spots = spots
    
    def spots(self, spots):
        self._spots = spots
        return self

    def get_time_step(self):
        return self._time_step
    
    def set_time_step(self, time_step):
        self._time_step = time_step
    
    def time_step(self, time_step):
        self._time_step = time_step
        return self

    def set_num_paths(self, num_paths):
        self._num_paths = num_paths

    def get_num_paths(self):
        return self._num_paths
    
    def num_paths(self, num_paths):
        self._num_paths = num_paths
        return self

    def set_num_steps(self, num_steps):
        self._num_steps = num_steps

    def get_num_steps(self):
        return self._num_steps

    def num_steps(self, num_steps):
        self._num_steps = num_steps
        return self

    def get_impvols(self):
        return self._impvols
    
    def set_impvols(self, impvols):
        self._impvols = impvols
    
    def impvols(self, impvols):
        self._impvols = impvols
        return self

    def load_json_data(self, json_: Union[str, dict]):
        """Load spot and implied vol surface data
        create _impvols rank 4 array from data
        dim: 0-strike
             1-maturity
             2-path
             3-step

        Args:
            json_ (Union[str, dict]): [description]
        """
        if isinstance(json_, str):
            dict_json = json.loads(json_)
        else:
            dict_json = json_
        path_step_shape = np.array(dict_json["Spot"]).shape
        if len(path_step_shape)==1:
            path_step_shape = (1,) + path_step_shape
        self._impvols = np.zeros((self._impvol_maturities.shape[0],self._impvol_strikes.shape[0])+path_step_shape) 
        self._spots = np.zeros(path_step_shape)
        for rf_name, rf_value in dict_json.items():
            if "Vol" in rf_name:
                m,k = parse_impvol_name(rf_name)
                self._impvols[np.where(self._str_impvol_maturities==float_to_str(m)),
                              np.where(self._str_impvol_strikes==float_to_str(k)),:,:] = np.array(rf_value)
            else:
                self._spots[:,:] = np.array(rf_value)
        self._num_paths = self._spots.shape[-2]
        self._num_steps = self._spots.shape[-1]
        self._impvols = tf.constant(self._impvols, dtype=np_dtype)
        self._spots = tf.constant(self._spots, dtype=np_dtype)
        self._time_step = dict_json["time_step_day"]/DAYS_PER_YEAR

    def get_spot(self, path_i: int, step_i: int=None):
        """Get the spot price for path_i and step_i

        Args:
            path_i (int): index of path
            step_i (int): index of step

        Returns:
            np_dtype: Spot price
        """
        if step_i is None:
            return self._spots[path_i, :]
        else:
            return self._spots[path_i, step_i]
    
    def get_var(self, path_i: int, step_i: int=None):
        """Return instantaneous var from Heston

        Args:
            path_i (int): index of path
            step_i (int, optional): index of step. Defaults to None.
        """
        if self._process_param["process_type"] != "Heston":
            return None
        if step_i is None:
            return self._vars[path_i, :]
        else:
            return self._vars[path_i, step_i]

    def get_impvol(self, path_i: int, step_i: int) -> ImpliedVolSurface:
        """Get the implied vol surface for path_i and step_i

        Args:
            path_i (int): index of path
            step_i (int): index of step

        Returns:
            ImpliedVolSurface: Implied Vol Surface at path_i and step_i
        """
        if self._process_param["process_type"] == "GBM":
            # loaded impvol
            vol_matrix = self._impvols
            backup_vol = self._process_param["param"]["vol"]
        else:
            # impvol already generated
            vol_matrix = self._impvols[:,:,path_i,step_i]
            backup_vol = self._vars[path_i, step_i]**0.5
        return ImpliedVolSurface(self._impvol_maturities, self._impvol_strikes, vol_matrix, 
                                 backup_vol=backup_vol)

    @classmethod
    def load_json(cls, json_: Union[str, dict]):
        if isinstance(json_, str):
            dict_json = json.loads(json_)
        else:
            dict_json = json_
        ret_equity_obj = cls().name(dict_json['name'])\
            .riskfactors(dict_json['riskfactors'])\
            .process_param(dict_json["process_param"])
        return ret_equity_obj

    def jsonify_dict(self) -> dict:
        dict_json = dict()
        dict_json["name"] = self._name
        dict_json["riskfactors"] = self._riskfactors
        dict_json["process_param"] = self._process_param
        return dict_json

    def __repr__(self):
        return json.dumps(self.jsonify_dict(), indent=4)
