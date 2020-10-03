"""Tests for the No Hedging bot."""

import acme
from acme import specs
from hb.bots.nohedgingbot.bot import NoHedgeBot
from hb.market_env.market import Market
from hb.market_env.market_test import MarketTest
from hb.market_env.portfolio import Portfolio
import unittest
import numpy as np
import matplotlib.pyplot as plt

class NoHedgeBotTest(unittest.TestCase):
    def test_gamma_nohedgebot(self):
        market = Market.load_market_file("Markets/Market_Example/spx_market.json")
        portfolio = Portfolio.load_portfolio_file("Markets/Market_Example/gamma_portfolio.json")
        self._set_up_greek_bot_test(market, portfolio)

    def test_regression_nohedgebot(self):
        market = MarketTest().set_up_regression_bsm_market()
        portfolio = Portfolio.make_portfolio(
            instruments=market.get_instruments([
                                                'AMZN', 
                                                # 'AMZN_OTC_1W_ATM_CALL',  
                                                # 'AMZN_OTC_1M_ATM_CALL', 
                                                'AMZN_OTC_3M_ATM_CALL'
                                                ]),
            holdings=[5., 
                      -10.],
            name="Three AMZN Calls"
        )
        self._set_up_greek_bot_test(market, portfolio, scenario='VaR')

    def _set_up_greek_bot_test(self, market, portfolio, scenario=None):
        market.set_portfolio(portfolio)
        if scenario:
            market.set_mode("scenario")
        else:
            market.set_mode("validation")
        spec = specs.make_environment_spec(market)
        
        # Construct the agent.
        agent = NoHedgeBot(
            environment_spec=spec
        )
        # Try running the environment loop. We have no assertions here because all
        # we care about is that the agent runs without raising any errors.
        loop = acme.EnvironmentLoop(market, agent)
        loop.run(num_episodes=market.get_total_episodes())
        predictor = agent.get_predictor()
        predictor._update_progress_figures()
        status = predictor._progress_measures
        pnl = predictor._pred_pnls
        import pandas as pd
        pd.DataFrame(pnl).to_csv('NoHedge_PnL.csv', index=False)
        print("No Hedge Bot PnL mean %s" % str(status['pnl_mean']))
        print("No Hedge Bot PnL std %s" % str(status['pnl_std']))
        print("No Hedge Bot 95VaR %s" % status['pnl_95VaR'])
        print("No Hedge Bot 99VaR %s" % status['pnl_99VaR'])
        print("No Hedge Bot 95CVaR %s" % status['pnl_95CVaR'])
        print("No Hedge Bot 99CVaR %s" % status['pnl_99CVaR'])

