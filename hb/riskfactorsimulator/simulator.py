from typing import List, Union
from hb.utils.process import *
import json

class Equity(object):
    __slots__ = ["_name", "_riskfactors", "_process_param"]

    def set_name(self, name):
        self._name = name
    
    def get_name(self):
        return self._name
    
    def name(self, name):
        self._name = name
        return self

    def set_riskfactors(self, riskfactors):
        self._riskfactors = riskfactors

    def get_riskfactors(self):
        return self._riskfactors

    def riskfactors(self, riskfactors):
        self._riskfactors = riskfactors
        return self

    def set_process_param(self, process_param):
        self._process_param = process_param

    def get_process_param(self, process_param):
        return self._process_param
    
    def process_param(self, process_param):
        self._process_param = process_param
        return self

    def get_implied_vol_surface_maturities(self):
        ...

    def get_implied_vol_surface_strikes(self):
        ...

    @classmethod
    def load_json(cls, json_: Union[str, dict]):
        if isinstance(json_, str):
            dict_json = json.loads(json_)
        else:
            dict_json = json_
        ret_equity_obj = cls().name(dict_json['name']).riskfactors(dict_json['riskfactors'])
        param_values = dict_json["process_param"]["param"]
        if dict_json["process_param"]["process_type"] == "GBM":
            ret_equity_obj.set_process_param(
                GBMProcessParam.load_json(param_values)
            )
        elif dict_json["process_param"]["process_type"] == "Heston":
            ret_equity_obj.set_process_param(
                HestonProcessParam.load_json(param_values)
            )
        return ret_equity_obj

    def jsonify_dict(self) -> dict:
        dict_json = dict()
        dict_json["name"] = self._name
        dict_json["riskfactors"] = self._riskfactors
        dict_json["process_param"] = dict() 
        if isinstance(self._process_param, GBMProcessParam):
            dict_json["process_param"]["process_type"] = "GBM"
        elif isinstance(self._process_param, HestonProcessParam):
            dict_json["process_param"]["process_type"] = "Heston"
        dict_json["process_param"]["param"] = self._process_param.jsonify_dict()
        return dict_json

    def __repr__(self):
        return json.dumps(self.jsonify_dict(), indent=4)


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
    __slots__ = ["_ir", "_equity", "_correlation"]

    def set_ir(self, ir: float):
        self._ir = ir

    def get_ir(self):
        return self._ir

    def ir(self, ir):
        self._ir = ir
        return self

    def set_equity(self, equity: List[Equity]):
        self._equity = equity

    def get_equity(self):
        return self._equity

    def equity(self, equity):
        self._equity = equity
        return self

    def set_correlation(self, correlation: List[Correlation]):
        self._correlation = correlation

    def get_correlation(self):
        return self._correlation

    def correlation(self, correlation: List[Correlation]):
        self._correlation = correlation
        return self

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
                "riskfactors": ["Spot", "Vol 3Mx100"],
                "process_param": {
                    "process_type": "GBM",
                    "param": {
                        "spot": 100,
                        "drift": 0.25,
                        "dividend": 0.0,
                        "vol": 0.3321
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
