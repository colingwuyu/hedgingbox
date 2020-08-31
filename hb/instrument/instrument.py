import abc
from hb.transaction_cost.transaction_cost import TransactionCost


class Instrument(abc.ABC):
    """Interface for instrument.
    """
    def __init__(self, name: str, tradable: bool, quote: float = None,
                 transaction_cost: TransactionCost = None,
                 underlying = None):
        self._name = name
        self._tradable = tradable
        self._quote = quote
        self._pricing_engine = None
        self._transaction_cost = transaction_cost
        self._underlying = underlying

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

    def set_pricing_engine(self, pricing_engine, *args):
        self._pricing_engine = pricing_engine
    
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
        return holding*self.price()

    def get_execute_cost(self, action: float) -> float:
        if self._tradable:
            return self._transaction_cost.execute(action)
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

    
