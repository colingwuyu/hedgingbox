import abc
import numpy as np
from hb.transaction_cost.transaction_cost import TransactionCost
from hb.utils.date import *
from os.path import join as pjoin
import os

class Instrument(abc.ABC):
    """Interface for instrument.
    """
    def __init__(self, name: str, 
                 tradable: bool,
                 transaction_cost: TransactionCost = None,
                 underlying = None):
        self._name = name
        self._tradable = tradable
        self._transaction_cost = transaction_cost
        self._str_underlying = None
        if isinstance(underlying, str):
            self._str_underlying = underlying
            self._underlying = None
        else:
            self._underlying = underlying
        self._price_cache = dict()
        self._exercised = False
        self._simulator_handler = None
        self._counter_handler = None

    def reset(self):
        self._price_cache = dict()
        self._exercised = False

    def get_name(self) -> str:
        return self._name

    def get_underlying_name(self) -> str:
        if self._underlying:
            return self._underlying.get_name()
        elif self._str_underlying:
            return self._str_underlying
        else:
            return ""

    def set_underlying(self, underlying):
        self._underlying = underlying

    def get_underlying(self):
        return self._underlying

    def underlying(self, underlying):
        self._underlying = underlying
        return self

    def get_is_tradable(self) -> bool:
        return self._tradable

    def set_simulator(self, simulator_handler, counter_handler):
        self._simulator_handler = simulator_handler
        self._counter_handler = counter_handler

    def exercise(self) -> float:
        self._exercised = True
        return 0.

    def get_price(self) -> float:
        """price of the instrument at path_i and step_i

        Returns:
            float: price
        """
        path_i, step_i = self._counter_handler.get_obj().get_path_step()
        if f"{path_i}_{step_i}" in self._price_cache:
            return self._price_cache[f"{path_i}_{step_i}"]
        else:
            _price = self._get_price(path_i, step_i)
            self._price_cache[f"{path_i}_{step_i}"] = _price 
            return _price

    @abc.abstractmethod
    def _get_price(self, path_i: int, step_i: int) -> float:
        """price of simulated episode at current time

        Returns:
            float: price
        """

    def get_market_value(self, holding: float) -> float:
        return holding*self.get_price()

    def get_execute_cost(self, action: float) -> float:
        if self._tradable:
            return self._transaction_cost.execute(self.get_market_value(action))
        else:
            return 0.0

    def get_delta(self) -> float:
        """delta of the instrument

        Returns:
            float: delta
        """
        return 1.

    def get_gamma(self) -> float:
        """gamma of the instrument

        Returns:
            float: gamma
        """
        return 0.

    def get_vega(self) -> float:
        """vega of the instrument

        Returns:
            float: vega
        """
        return 0.

    def get_maturity_time(self) -> float:
        """maturity time of the instrument from inception

        Returns:
            float: maturity time
        """
        return 0.0

    def get_remaining_time(self) -> float:
        """remaining time of the instrument

        Returns:
            float: remaining time
        """
        return np.infty

    def get_delivery_amount(self) -> float:
        """after expiry, how much amount is delivered

        Returns:
            float: delivery shares for option exercise
        """
        return 0

    def get_receive_amount(self) -> float:
        """after expiry, how much amount is received

        Returns:
            float: receive cahs for option exercise
        """
        return 0

    def get_is_physical_settle(self) -> bool:
        return True

    def get_is_expired(self) -> bool:
        """check if instrument expires

        Returns:
            bool: True if it expires, False otherwise
        """
        return True

