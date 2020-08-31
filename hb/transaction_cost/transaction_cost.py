import abc


class TransactionCost:
    @abc.abstractmethod
    def execute(self, action: float, *args) -> float:
        """transaction cost of excuting action 

        Args:
            action (float): buy/sell amount

        Returns:
            float: total transaction cost
        """