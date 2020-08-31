from hb.transaction_cost.transaction_cost import TransactionCost


class PercentageTransactionCost(TransactionCost):
    def __init__(self, percentage_cost: float):
        """Transaction cost that is percentage of buy/sell amount

        Args:
            percentage_cost (float): transaction cost percentage
        """ 
        self._percentage_cost = percentage_cost
        super().__init__()

    def execute(self, action, *args):
        price = args[0]
        return action*price*self._percentage_cost

    def __repr__(self):
        return f'percentage transaction cost({self._percentage_cost})'
