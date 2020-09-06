import abc
from hb.transaction_cost.transaction_cost import TransactionCost


class Instrument(abc.ABC):
    """Interface for instrument.
    """
    def __init__(self, name: str, tradable: bool, quote: float = None,
                 transaction_cost: TransactionCost = None,
                 underlying = None, trading_limit: float = 1e10):
        self._name = name
        self._tradable = tradable
        self._quote = quote
        self._pricing_engine = None
        self._transaction_cost = transaction_cost
        self._underlying = underlying
        self._trading_limit = trading_limit

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

    @abc.abstractmethod
    def get_price(self, *args) -> float:
        """price of the instrument

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
