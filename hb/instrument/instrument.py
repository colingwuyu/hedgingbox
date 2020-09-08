import abc
import pandas as pd
from hb.transaction_cost.transaction_cost import TransactionCost
from hb.utils.date import *
from os.path import join as pjoin
import os

class Instrument(abc.ABC):
    """Interface for instrument.
    """
    def __init__(self, name: str, tradable: bool, quote: float = None,
                 transaction_cost: TransactionCost = None,
                 underlying = None, trading_limit: float = 1e10,
                 pred_episodes=1_000):
        self._name = name
        self._tradable = tradable
        self._quote = quote
        self._pricing_engine = None
        self._transaction_cost = transaction_cost
        self._underlying = underlying
        self._trading_limit = trading_limit
        self._pred_episodes = pred_episodes
        self._pred_mode = False
        self._pred_price_path = None
        self._dir = None
        self._cur_pred_path = -1
        self._cur_pred_file = None
        self._cur_price = (0., None)
        self._num_steps = None

    def get_name(self) -> str:
        return self._name

    def get_underlying_name(self) -> str:
        if self._underlying:
            return self._underlying.get_name()
        else:
            return ''

    def get_quote(self) -> float:
        return self._quote
    
    def set_quote(self, quote: float):
        self._quote = quote
    
    def set_num_steps(self, num_steps: int):
        """Set number of steps per episode

        Args:
            num_step (int): Number of steps per episode
        """
        self._num_steps = num_steps

    def get_num_steps(self):
        return self._num_steps

    def set_pred_episodes(self, pred_episodes: int):
        """Set number of episodes for prediction

        Args:
            pred_episodes (int): Number of episodes for prediction
        """
        self._pred_episodes = pred_episodes
    
    def get_pred_episodes(self):
        return self._pred_episodes

    def set_pred_mode(self, pred_mode: bool):
        """Set prediction mode

        Args:
            pred_mode (bool): True - set to use prediction price paths
        """
        self._pred_mode = pred_mode
        if self._cur_pred_file:
            # use load pred episodes
            self._cur_pred_path = -1
        else:
            # use sim for pred episodes
            self._cur_pred_path = None

    def load_pred_episodes(self, pred_file: str='pred_price.csv', *args):
        """Load prediction episodes into memory if it was saved in files
           to be inherited, if intends to load more pred attributes other than price
        Args:
            pred_file (str, optional): The prediction saved files. Defaults to 'pred_price.csv'.
        """
        if os.path.exists(pjoin(self.get_pred_dir(), pred_file)):
            self._cur_pred_file = pred_file
            self._pred_price_path = pd.read_csv(pjoin(self.get_pred_dir(), pred_file)).values
            self.set_pred_episodes(self._pred_price_path.shape[0])
        else:
            self._cur_pred_file = None
            if self._underlying is not None:
                self.set_pred_episodes(self._underlying.get_pred_episodes())
            self._pred_price_path = [[]]

    def save_pred_episodes(self, file_name: str='pred_price.csv'):
        """Save prediction episodes into files
           to be inherited, if intends to save more pred attributes other than price
        """
        if self._cur_pred_file is None:
            pd.DataFrame(self._pred_price_path).to_csv(pjoin(self.get_pred_dir(), file_name), index=False)
            self._cur_pred_file = file_name

    def set_portfolio_dir(self, portfolio_dir):
        """Set portfolio directory for saving data

        Args:
            portfolio_dir (str): Directory of portfolio
        """
        self._dir = pjoin(portfolio_dir, "Instrument_"+self._name)
        if not os.path.exists(self._dir):
            os.makedirs(self._dir)
        if not os.path.exists(self.get_pred_dir()):
            os.makedirs(self.get_pred_dir())
        self.load_pred_episodes()

    def get_dir(self):
        """Generate the instrument directory

        Returns:
            [str]: instrument directory 
        """
        return self._dir
    
    def get_pred_dir(self):
        """Generate the prediction directory

        Returns:
            [str]: instrument prediction directory
        """
        return pjoin(self._dir, 'Pred')

    def quote(self, quote: float):
        self._quote = quote
        return self

    def set_underlying(self, underlying):
        self._underlying = underlying

    def underlying(self, underlying):
        self._underlying = underlying
        return self

    def get_is_tradable(self) -> bool:
        return self._tradable

    @abc.abstractmethod
    def set_pricing_engine(self, *args):
        """set pricing engine
        """
      
    def pricing_engine(self, pricing_engine, *args):
        self._pricing_engine = pricing_engine
        return self

    def get_price(self, *args) -> float:
        """price of the instrument at current time

        Returns:
            float: price
        """
        if (abs(self._cur_price[0] - get_cur_time())<1e-5) \
            and (self._cur_price[1] is not None):
            # already has cached price for current time step
            return self._cur_price[1]
        else:
            # first visit at time step, need reprice
            self._cur_price = (get_cur_time(), None)
            # generate price
            if self._pred_mode:
                price = self.get_pred_price()
            else:
                price = self.get_sim_price()
            # cache price for current time step
            self._cur_price = (self._cur_price[0], price)
            return price

    @abc.abstractmethod
    def get_sim_price(self, *args) -> float:
        """price of simulated episode at current time

        Returns:
            float: price
        """

    @abc.abstractmethod
    def get_pred_price(self, *args) -> float:
        """price of prediction episode at current time
        Returns:
            float: price
        """

    def get_market_value(self, holding: float) -> float:
        return holding*self.get_price()

    def get_execute_cost(self, action: float) -> float:
        if self._tradable:
            return self._transaction_cost.execute(action, 
                                                  self.get_market_value(action))
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

    @abc.abstractmethod
    def get_maturity_time(self) -> float:
        """maturity time of the instrument

        Returns:
            float: maturity time
        """

    @abc.abstractmethod
    def get_remaining_time(self) -> float:
        """remaining time of the instrument

        Returns:
            float: remaining time
        """

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
        return False

    def get_is_exercised(self) -> bool:
        """after expiry, if the derivative is exercised

        Returns:
            bool: true if exercised
        """
        return True

    def get_is_expired(self) -> bool:
        """check if instrument expires

        Returns:
            bool: True if it expires, False otherwise
        """
        return True

    def get_trading_limit(self) -> float:
        """trading block limit

        Returns:
            float: the maximum shares one buy/sell action can be executed 
                   None - means no limit
        """
        return self._trading_limit
