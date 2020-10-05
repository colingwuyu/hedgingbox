import json
import numpy as np
from typing import Union


class RiskLimits():
    __slots__ = ["_delta", "_gamma", "_vega"]

    def get_delta(self):
        return self._delta
    
    def set_delta(self, delta):
        self._delta = delta

    def delta(self, delta):
        self._delta = delta
        return self

    def get_gamma(self):
        return self._gamma
    
    def set_gamma(self, gamma):
        self._gamma = gamma

    def gamma(self, gamma):
        self._gamma = gamma
        return self
 
    def get_vega(self):
        return self._vega
    
    def set_vega(self, vega):
        self._vega = vega

    def vega(self, vega):
        self._vega = vega
        return self

    @classmethod
    def load_json(cls, json_: Union[dict, str]):
        if isinstance(json_, str):
            dict_json = json_.loads(json_)
        else:
            dict_json = json_
        ret_risk_limits = cls()
        if "delta" in dict_json:
            ret_risk_limits.set_delta(dict_json["delta"])
        if "gamma" in dict_json:
            ret_risk_limits.set_delta(dict_json["gamma"])
        if "vega" in dict_json:
            ret_risk_limits.set_delta(dict_json["vega"])
        return ret_risk_limits
    
    def jsonify_dict(self) -> dict:
        dict_json = dict()
        if hasattr(self, "_delta"):
            dict_json["delta"] == self._delta
        if hasattr(self, "_gamma"):
            dict_json["gamma"] == self._gamma
        if hasattr(self, "_vega"):
            dict_json["vega"] == self._vega
        return dict_json

    def __repr__(self):
        return json.dumps(self.jsonify_dict(), indent=4)            

    def review_actions(self, actions, portfolio):
        """Review the actions, if actions breach risk limist, then truncate the actions

        Args:
            actions (np.ndarray): trading action
            portfolio (Portfolio): portfolio
        """
        if hasattr(self, "_delta"):
            total_delta = 0
            inc_delta = np.zeros(actions.shape)
            deltas = np.zeros(actions.shape)
            hedgings = portfolio.get_hedging_portfolio()
            positions = portfolio.get_portfolio_positions()
            trunc_actions = np.zeros(actions.shape)
            for position in positions:
                total_delta += position.get_instrument().get_delta()*position.get_holding()
            for i, action in enumerate(actions):
                inc_delta[i] = hedgings[i].get_instrument().get_delta()*action
                deltas[i] = hedgings[i].get_instrument().get_delta()
            ind = np.argsort(inc_delta)
            delta_ind = np.argsort(deltas)
            low_i = 0; up_i = len(ind)
            total_delta_inc = inc_delta.sum()
            if ((total_delta_inc + total_delta) < self._delta[0]) and (total_delta < self._delta[0]) and (total_delta_inc > 0):
                trunc_actions = actions
            elif ((total_delta_inc + total_delta) > self._delta[1]) and (total_delta > self._delta[1]) and (total_delta_inc < 0):
                trunc_actions = actions
            else:
                while (((total_delta_inc + total_delta) < self._delta[0]) or \
                    ((total_delta_inc + total_delta) > self._delta[1])) and \
                    (up_i > low_i):    
                    if (total_delta_inc + total_delta) < self._delta[0]:
                        # exceeds lower limit
                        low_i += 1
                        total_delta_inc = inc_delta[ind[low_i:]].sum()
                    elif (total_delta_inc + total_delta) > self._delta[1]:
                        # exceeds upper limit
                        up_i -= 1
                        total_delta_inc = inc_delta[ind[:up_i]].sum()
                trunc_actions[ind[low_i:up_i]] = actions[ind[low_i:up_i]].copy()
                if (low_i != 0):
                    # exceeds lower limit
                    diff_delta = self._delta[0] - (inc_delta[ind[low_i:]].sum() + total_delta)
                    trunc_actions[delta_ind[-1]] += diff_delta/deltas[delta_ind[-1]]
                elif up_i != len(ind):
                    # exceeds upper limit
                    diff_delta = self._delta[1] - (inc_delta[ind[:up_i]].sum() + total_delta)
                    trunc_actions[delta_ind[-1]] = diff_delta/deltas[delta_ind[-1]]
        else:
            trunc_actions = actions.copy()
        # print(total_delta, total_delta_inc, actions, trunc_actions)
        return trunc_actions

