from typing import List
from hb.instrument.instrument import Instrument


class Portfolio():
    def __init__(self, 
                 instruments: List[Instrument], 
                 holdings: List[float]):
        self._instruments = instruments
        self._init_holdings = holdings.copy()
        self._holdings = holdings.copy()

    def get_nav(self):
        nav = 0.
        for i, h in zip(self._instruments, self._holdings):
            nav += i.get_market_value(h)
        return nav

    def rebalance(self, actions) -> float:
        cashflow = 0.
        action_i = 0
        rebalance_cost = 0.
        for i, instrument in enumerate(self._instruments):
            if instrument.get_is_tradable():
                action = actions[i]
                self._holdings[i] += action
                proceeds = instrument.get_market_value(action) - instrument.get_execute_cost(action)
                cashflow += proceeds
                rebalance_cost_i = -instrument.get_execute_cost(actions[i])
                rebalance_cost += rebalance_cost_i
        return cashflow, rebalance_cost
    
    def dump_cost(self) -> float:
        cost = 0.
        for i, instrument in enumerate(self._instruments):
            if instrument.get_is_tradable():
                cost_i = -instrument.get_execute_cost(self._holdings[i])
                cost += cost_i
        return cost

    def get_rebalance_cost(self, actions):
        rebalance_cost = 0.
        for i, instrument in enumerate(self._instruments):
            if instrument.get_is_tradable():
                rebalance_cost_i = -instrument.get_execute_cost(actions[i])
                rebalance_cost += rebalance_cost_i
        return rebalance_cost

    def get_instruments(self):
        return self._instruments

    def get_holdings(self):
        return self._holdings
    
    def reset(self):
        self._holdings = self._init_holdings.copy()
