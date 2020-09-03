from typing import Union, List
from hb.utils.termstructure import *
from hb.utils.date import *
import QuantLib as ql
import math

class CashAccount(object):
    def __init__(self, 
                 times: Union[float,List[float]] = 10.,
                 interest_rates: Union[float,List[float]] =0.01):
        """Cash Account
        An account accumulates interests. 
        The account balance can be either positive or negative
        positive balance earns interests 
        negative balance pays interests

        TODO: expand static interest rate to a dynamic model
        Args:
            balance (float): account balance
            interest_rate (float): interest rate
        """
        self._balance = 0.
        if type(times) == float: 
            rate_ts = create_flat_forward_ts(interest_rates)
        else:
            rate_ts = create_zero_curve_ts(times, interest_rates)
        self._rate_ts = rate_ts
        self._cur_time = 0.0

    def set_initial_balance(self, balance: float):
        self._initial_balance = balance
        self._balance = balance

    def add(self, cash: float):
        """deposit/withdraw cash to account

        Args:
            cash (float): deposit/withdraw amounts (value to be positive/negative)
        """ 
        self._balance += cash

    def get_balance(self) -> float:
        return self._balance

    def accrue_interest(self):
        cur_time = get_cur_time()
        if cur_time > self._cur_time:
            # move forward one step
            # accrue interest
            interest_rate = self._rate_ts.forwardRate(self._cur_time, cur_time, ql.Continuous).rate()
        else:
            interest_rate = 0.
        self._cur_time = cur_time
        return math.exp(interest_rate*(cur_time - self._cur_time))

    def reset(self):
        self._balance = 0.
        self._cur_time = get_cur_time()

