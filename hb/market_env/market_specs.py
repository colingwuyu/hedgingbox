from dm_env import specs
import numpy as np
from collections import namedtuple

MarketEnvParam = namedtuple('MarketEnvParam',
                            'stock_ticker_size '
                            'stock_price_lower_bound '
                            'stock_price_upper_bound '
                            'lot_size '
                            'buy_sell_lots_bound '
                            'holding_lots_bound')


class DiscretizedBoundedArray(specs.BoundedArray):

    __slots__ = ('_discretize_step')
    __hash__ = None

    def __init__(self, shape, dtype, minimum, maximum, discretize_step, name=None):
        super(DiscretizedBoundedArray, self).__init__(
            shape, dtype, minimum, maximum, name)
        self._discretize_step = discretize_step

    @property
    def discretize_step(self):
        return self._discretize_step


class StockMarketObservationSpec(specs.BoundedArray):
    def __init__(self, shape, dtype, obs_attr, ticker_size, stock_price_bound,
                 lot_size, holding_lots_bound, name=None):
        stock_price_ind = obs_attr.index('stock_price')
        holding_ind = obs_attr.index('stock_holding')
        minimum = [-np.infty]*shape[0]
        maximum = [np.infty]*shape[0]
        minimum[stock_price_ind] = ticker_size
        maximum[stock_price_ind] = stock_price_bound
        minimum[holding_ind] = -holding_lots_bound*lot_size
        maximum[holding_ind] = holding_lots_bound*lot_size
        super(StockMarketObservationSpec, self).__init__(shape, dtype, minimum,
                                                         maximum, name)
        self._ticker_size = ticker_size
        self._stock_price_bound = stock_price_bound
        self._lot_size = lot_size
        self._min_holding = minimum[holding_ind]
        self._max_holding = maximum[holding_ind]

    def price_grid(self):
        if (self._stock_price_bound != np.infty):
            return np.arange(self._ticker_size, self._stock_price_bound + self._ticker_size)
        else:
            return None

    def holding_grid(self):
        if (self._min_holding != -np.infty) and (self._max_holding != np.infty):
            return np.arange(self._min_holding, self._max_holding, self._lot_size)
        else:
            return None


class StockMarketActionSpec(specs.BoundedArray):
    def __init__(self, shape, dtype,
                 lot_size, buy_sell_lots_bound, name=None):
        minimum = [-lot_size*buy_sell_lots_bound]
        maximum = [lot_size*buy_sell_lots_bound]
        super(StockMarketActionSpec, self).__init__(shape, dtype, minimum,
                                                    maximum, name)
        self._lot_size = lot_size
        self._min_buy_sell = minimum[0]
        self._max_buy_sell = maximum[0]

    def action_grid(self):
        if (self._min_buy_sell != np.infty) and (self._max_buy_sell != np.infty):
            return np.arange(self._min_buy_sell, self._max_buy_sell + self._lot_size, self._lot_size)
        else:
            return None
