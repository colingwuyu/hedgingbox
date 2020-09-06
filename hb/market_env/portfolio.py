from typing import List
from hb.instrument.instrument import Instrument
from hb.instrument.stock import Stock

class Position():
    def __init__(self, instrument, holding=0.):
        self._instrument = instrument
        self._holding = holding
        self._init_holding = holding

    def reset(self):
        self._holding = self._init_holding

    def get_instrument(self):
        return self._instrument

    def get_holding(self):
        return self._holding

    def get_init_holding(self):
        return self._init_holding

    def set_init_holding(self, init_holding):
        self._init_holding = init_holding

    def buy(self, shares: float):
        """buy shares

        Args:
            shares (float): if it is positive, then means to buy shares
                            if it is negative, then means to sell shares

        Returns:
            cashflow (float):   the proceed cash flow including transaction cost 
                                it is positive cashflow if sell shares
                                it is negative cashflow if buy shares
                                transaction cost is always negative cashflow
            transaction cost (float): the transaction cost for buying shares
        """
        self._holding += shares
        trans_cost = self._instrument.get_execute_cost(shares)
        return - self._instrument.get_market_value(shares) - trans_cost, trans_cost

    def get_market_value(self):
        return self._instrument.get_market_value(self._holding)


class Portfolio():
    def __init__(self, 
                 positions: List[Position]):
        self._positions = positions
        self._hedging_portfolio = []
        self._liability_portfolio = []
        for position in positions:
            if position.get_instrument().get_is_tradable():
                self._hedging_portfolio += [position]
            else:
                self._liability_portfolio += [position]

    @classmethod
    def make_portfolio(cls,
                       instruments: List[Instrument],
                       holdings: List[float]):
        positions = []
        for instrument, holding in zip(instruments, holdings):
            positions += [Position(instrument, holding)]
        return cls(positions)

    def get_hedging_portfolio(self):
        return self._hedging_portfolio

    def get_liability_portfolio(self):
        return self._liability_portfolio

    def get_portfolio_positions(self):
        return self._positions

    def reset(self):
        for position in self._positions:
            position.reset()

    def get_nav(self):
        nav = 0.
        for position in self._positions:
            nav += position.get_market_value()
        return nav

    def rebalance(self, actions):
        """Rebalance portfolio with hedging actions
           Also deal with the expiry events
           When a derivative expires:
            reduce the hedging positions' transaction cost according to the exercise
                - if cash delivery: include the corresponding transaction costs for dumping the hedging positions
                - if physical delivery: no transaction costs for the hedging position delivery 

        Args:
            actions ([float]): hedging buy/sell action applied to hedging portfolio

        Returns:
            cashflow [float]: step cashflow (buy/sell proceeds, and option exercise payoff go to cash account)
            trans_cost [float]: transaction cost at the time step
        """
        cashflows = 0.
        trans_costs = 0.
        for action, hedging_position in zip(actions, self._hedging_portfolio):
            # rebalance hedging positions
            proceeds, trans_cost = hedging_position.buy(action)
            cashflows += proceeds
            trans_costs += trans_cost
            
        for derivative in self._liability_portfolio:
            if abs(derivative.get_instrument().get_remaining_time()-0.0) < 1e-5:
                # position expired
                derivative_cash_flow, rebate_trans_cost = self.derivative_exercise_event(derivative)
                # derivative payoff is paid cashflow
                cashflows += derivative_cash_flow
                # the hedging action includes the delivery or dumping due to derivative gets exercised
                # if the derivative is physically delivered, the transaction cost should exclude the hedging_position delivery action
                # in such case rebate trans_costs to the delivery shares
                trans_costs -= rebate_trans_cost
        return cashflows, trans_costs
    
    def derivative_exercise_event(self, derivative_position: Position):
        """Deal derivative exercise event
            When a derivative expires:
            reduce the hedging positions according to the delivery shares of exercise
                - if cash delivery: include the corresponding transaction costs for dumping the hedging positions
                - if physical delivery: no extra transaction costs 

        Returns:
            cashflow [float]: proceeds due to dumping hedging positions 
        """
        cf = 0.
        trans_cost = 0.
        shares = derivative_position.get_instrument().get_delivery_amount()
        dump_shares = derivative_position.get_holding()*shares
        if not derivative_position.get_instrument().get_is_physical_settle():
            # physical settle
            trans_cost = derivative_position.get_instrument().get_execute_cost(dump_shares)
        cashflow = derivative_position.get_market_value()
        # derivative position cleared
        derivative_position.buy(-derivative_position.get_holding())
        return cashflow, trans_cost


    def dump_portfolio(self):
        """Dump portfolio at terminal step

        Returns:
            cashflow [float]: terminal cashflow (dump hedging portfolio proceeds, and derivative exercise payoff go to cash account)
            trans_cost [float]: transaction cost for dumping portfolio at terminal cashflow
        """
        cashflows = 0.
        trans_costs = 0.
        for derivative in self._liability_portfolio:
            if abs(derivative.get_instrument().get_remaining_time()-0.0) < 1e-5:    
                derivative_cash_flow, rebate_trans_cost = self.derivative_exercise_event(derivative)
                trans_costs -= rebate_trans_cost
                cashflows += derivative_cash_flow
        for hedging_position in self._hedging_portfolio:
            # rebalance hedging positions
            proceeds, trans_cost = hedging_position.buy(-hedging_position.get_holding())
            cashflows += proceeds
            trans_costs += trans_cost
        return cashflows, trans_costs
