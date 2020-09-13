from hb.instrument.stock import Stock
from hb.instrument.european_option import EuropeanOption
from hb.instrument.variance_swap import VarianceSwap
from hb.transaction_cost.percentage_transaction_cost import PercentageTransactionCost
from hb.utils import date as date_util


class InstrumentFactory():
    @classmethod
    def create(cls, str_instrument: str):
        """Create instrument

        Args:
            str_instrument (str): 
                Stock:
                    'Stock Ticker Quote annual_yield% dividend_yield% transaction_cost%'  
                European Option:
                    'EuroOpt Ticker OTC/Listed Maturity Call/Put Strike IV% transaction_cost% (ShortName)'
                Variance Swap:
                    'VarSwap Ticker Maturity VolStrike% Alpha Notional (ShortName)'
        """
        params = str_instrument.split(' ')
        if params[0] == 'Stock':
            return Stock(
                name=params[1],
                quote=float(params[2]),
                annual_yield=float(params[3])/100.,
                dividend_yield=float(params[4])/100.,
                transaction_cost=PercentageTransactionCost(float(params[5])/100.)
            )
        if params[0] == 'EuroOpt':
            return EuropeanOption(
                name=params[-1][1:-1],
                option_type=params[4],
                maturity=date_util.get_period_from_str(params[3]),
                strike=float(params[5]),
                tradable=True if params[2]=='Listed' else False,
                quote=float(params[6])/100.,
                transaction_cost=PercentageTransactionCost(float(params[7])/100.)
            )
        if params[0] == 'VarSwap':
            return VarianceSwap(
                name=params[-1][1:-1],
                vol_strike=float(params[3])/100.,
                maturity=date_util.get_period_from_str(params[2]),
                alpha=float(params[4]),
                notional=float(params[5])
            )
        return None

if __name__ == "__main__":
    spx = InstrumentFactory.create(
        'Stock AMZN 3400 25 0 0.15'
    )
    spx_3m = InstrumentFactory.create(
        'EuroOpt AMZN Listed 1W Call 3400 33.21 5 (AMZN_Call0)'
    ).underlying(spx)
    print(spx)
    print(spx_3m)
        