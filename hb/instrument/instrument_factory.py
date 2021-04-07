from hb.instrument.stock import Stock
from hb.instrument.european_option import EuropeanOption
from hb.instrument.variance_swap import VarianceSwap
from hb.transaction_cost.percentage_transaction_cost import PercentageTransactionCost
from hb.utils import date as date_util
import numpy as np


class InstrumentFactory():
    @staticmethod
    def create(str_instrument: str):
        """Create instrument

        Args:
            str_instrument (str): 
                Stock:
                    'Stock Ticker annual_yield% dividend_yield% transaction_cost% daily_volume mi_alpha'  
                European Option:
                    'EuroOpt Ticker OTC/Listed Maturity Call/Put Strike transaction_cost% (ShortName)'
                Variance Swap:
                    'VarSwap Ticker Maturity VolStrike VarNotional (ShortName)'
        """
        params = str_instrument.split(' ')
        if params[0] == 'Stock':
            daily_volume = np.infty
            mi_alpha = 1.0
            if len(params) >= 6:
                daily_volume = float(params[5])
                if len(params) == 7:
                    mi_alpha = float(params[6])
            return Stock(
                name=params[1],
                annual_yield=float(params[2])/100.,
                dividend_yield=float(params[3])/100.,
                transaction_cost=PercentageTransactionCost(float(params[4])/100.),
                daily_volume=daily_volume, mi_alpha=mi_alpha
            )
        if params[0] == 'EuroOpt':
            if "-" not in params[3]:
                maturity = date_util.get_period_from_str(params[3])  
            else:
                maturity = date_util.time_between(date_util.date_from_str(params[3]))
            return EuropeanOption(
                name=params[-1][1:-1],
                underlying=params[1],
                option_type=params[4],
                maturity=maturity,
                strike=float(params[5]),
                tradable=True if params[2]=='Listed' else False,
                transaction_cost=PercentageTransactionCost(float(params[6])/100.)
            )
        if params[0] == 'VarSwap':
            return VarianceSwap(
                name=params[-1][1:-1],
                underlying=params[1],
                vol_strike=float(params[3]),
                maturity=date_util.get_period_from_str(params[2]),
                var_notional=float(params[4])
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
        