"""Tests for the Delta Hedging bot."""

import acme
from acme import specs
from hb.bots import euro_deltabot
from hb.market_env.market_test import MarketTest
from hb.market_env.portfolio import Portfolio
import unittest
import numpy as np
import matplotlib.pyplot as plt


class DeltaBotTest(unittest.TestCase):
    def test_regression_deltabot(self):
        market = MarketTest().set_up_regression_bsm_market()
        portfolio = Portfolio.make_portfolio(
            instruments=market.get_instruments([
                                                'AMZN', 
                                                'AMZN_OTC_1W_ATM_CALL',  
                                                'AMZN_OTC_1M_ATM_CALL', 
                                                'AMZN_OTC_3M_ATM_CALL'
                                                ]),
            holdings=[0., 
                      -10., 
                      -10., 
                      -10.]
        )
        self._set_up_delta_bot_test(market, portfolio)

    def test_deltabot_with_heston_amzn(self):
        # Create a GBM market
        market = MarketTest().set_up_heston_market()
        portfolio = Portfolio.make_portfolio(
            instruments=market.get_instruments([
                                                'AMZN', 
                                                'AMZN_OTC_1W_ATM_CALL',  
                                                'AMZN_OTC_1M_ATM_CALL', 
                                                'AMZN_OTC_3M_ATM_CALL'
                                                ]),
            holdings=[0., 
                      -10., 
                      -10., 
                      -10.]
        )
        self._set_up_delta_bot_test(market, portfolio)

    def _set_up_delta_bot_test(self, market, portfolio):
        market.init_portfolio(portfolio)
        spec = specs.make_environment_spec(market)
        
        # Construct the agent.
        agent = euro_deltabot.DeltaHedgeBot(
            portfolio=portfolio,
            environment_spec=spec
        )
        market.set_pred_mode(True)
        # Try running the environment loop. We have no assertions here because all
        # we care about is that the agent runs without raising any errors.
        loop = acme.EnvironmentLoop(market, agent)
        loop.run(num_episodes=1_000)
        agent._predictor._update_progress_figures()
        status = agent._predictor._progress_measures
        print("Delta Bot PnL mean %s" % str(status['pnl_mean']))
        print("Delta Bot PnL std %s" % str(status['pnl_std']))
        print("Delta Bot 95VaR %s" % status['pnl_95VaR'])
        print("Delta Bot 99VaR %s" % status['pnl_99VaR'])
        print("Delta Bot 95CVaR %s" % status['pnl_95CVaR'])
        print("Delta Bot 99CVaR %s" % status['pnl_99CVaR'])


if __name__ == '__main__':
    # unittest.main()
    DeltaBotTest().test_deltabot()
