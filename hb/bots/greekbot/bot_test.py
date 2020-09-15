"""Tests for the Delta Hedging bot."""

import acme
from acme import specs
from hb.bots.greekbot.bot import GreekHedgeBot 
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
            holdings=[4.5, 
                      -3., 
                      -3., 
                      -3.],
            name="Three AMZN Calls"
        )
        self._set_up_greek_bot_test(market, portfolio)

    def test_bs_deltabot_with_heston_amzn(self):
        # Create a GBM market
        market = MarketTest().set_up_heston_market()
        portfolio = Portfolio.make_portfolio(
            instruments=market.get_instruments([
                                                'AMZN', 
                                                # 'AMZN_OTC_1W_ATM_CALL',  
                                                # 'AMZN_OTC_1M_ATM_CALL', 
                                                'AMZN_OTC_3M_ATM_CALL'
                                                ]),
            holdings=[0., 
                    #   -10., 
                    #   -10., 
                      -10.],
            name="AMZN and AMZN_3M_Call"
        )
        self._set_up_greek_bot_test(market, portfolio)

    def test_bs_deltabot_with_heston_spx(self):
        # Create a GBM market
        market = MarketTest().set_up_heston_market()
        portfolio = Portfolio.make_portfolio(
            instruments=market.get_instruments([
                                                'SPX', 
                                                'SPX_OTC_1W_ATM_CALL',  
                                                # 'SPX_OTC_1M_ATM_CALL', 
                                                # 'SPX_OTC_3M_ATM_CALL'
                                                ]),
            holdings=[0., 
                    #   -10., 
                    #   -10., 
                      -10.],
            name="SPX and SPX_1W_Call"
        )
        self._set_up_greek_bot_test(market, portfolio)

    def test_heston_deltabot_with_heston_amzn(self):
        # Create a GBM market
        market = MarketTest().set_up_heston_market()
        portfolio = Portfolio.make_portfolio(
            instruments=market.get_instruments([
                                                'AMZN', 
                                                'AMZN_OTC_1W_ATM_CALL',  
                                                # 'AMZN_OTC_1M_ATM_CALL', 
                                                # 'AMZN_OTC_3M_ATM_CALL'
                                                ]),
            holdings=[0., 
                    #   -10., 
                    #   -10., 
                      -10.],
            name="AMZN and AMZN_1W_Call"
        )
        self._set_up_greek_bot_test(market, portfolio, use_bs_delta=False)

    def test_heston_deltabot_with_heston_spx(self):
        # Create a GBM market
        market = MarketTest().set_up_heston_market()
        portfolio = Portfolio.make_portfolio(
            instruments=market.get_instruments([
                                                'SPX', 
                                                'SPX_OTC_1W_ATM_CALL'
                                                ]),
            holdings=[0., 
                      -10.],
            name="SPX and SPX_1W_Call"
        )
        self._set_up_greek_bot_test(market, portfolio, use_bs_delta=False)

    def test_deltabot_with_bsm_amzn(self):
        # Create a GBM market
        market = MarketTest().set_up_bsm_market()
        portfolio = Portfolio.make_portfolio(
            instruments=market.get_instruments([
                                                'AMZN', 
                                                'AMZN_OTC_1W_ATM_CALL'
                                                ]),
            holdings=[0.,
                      -10.],
            name="AMZN and AMZN_1W_Call"
        )
        self._set_up_greek_bot_test(market, portfolio)

    def test_deltabot_with_bsm_spx_covid19(self):
        # Create a GBM market
        market = MarketTest().set_up_regression_bsm_market()
        portfolio = Portfolio.make_portfolio(
            instruments=market.get_instruments([
                                                'AMZN', 
                                                'AMZN_OTC_3M_ATM_CALL'
                                                ]),
            holdings=[0.,
                      -10.],
            name="Three AMZN Calls"
        )
        self._set_up_greek_bot_test(market, portfolio, scenario='Covid19')
    
    def test_deltabot_with_bsm_spx_var(self):
        # Create a GBM market
        market = MarketTest().set_up_regression_bsm_market()
        portfolio = Portfolio.make_portfolio(
            instruments=market.get_instruments([
                                                'AMZN', 
                                                'AMZN_OTC_3M_ATM_CALL'
                                                ]),
            holdings=[0.,
                      -10.],
            name="Three AMZN Calls"
        )
        self._set_up_greek_bot_test(market, portfolio, scenario='VaR')
    
    def test_deltabot_with_bsm_spx_svar(self):
        # Create a GBM market
        market = MarketTest().set_up_regression_bsm_market()
        portfolio = Portfolio.make_portfolio(
            instruments=market.get_instruments([
                                                'AMZN', 
                                                'AMZN_OTC_3M_ATM_CALL'
                                                ]),
            holdings=[0.,
                      -10.],
            name="Three AMZN Calls"
        )
        self._set_up_greek_bot_test(market, portfolio, scenario='SVaR')

    def test_variance_swap_hedge_heston(self):
        market = MarketTest().set_up_heston_market()
        variance_swap_portfolio_instruments = market.get_instrument('SPX_3M_VAR_SWAP').get_hedging_instrument_names() + ['SPX_3M_VAR_SWAP']
        holdings = [0]*len(variance_swap_portfolio_instruments)
        holdings[-1] = -1
        portfolio = Portfolio.make_portfolio(
            instruments=market.get_instruments(variance_swap_portfolio_instruments),
            holdings=holdings,
            name="SPX 3M Variance Swap"
        )
        self._set_up_greek_bot_test(market, portfolio)

    def test_deltabot_with_bsm_spx_amzn(self):
        # Create a GBM market
        market = MarketTest().set_up_bsm_market()
        portfolio = Portfolio.make_portfolio(
            instruments=market.get_instruments([
                                                'AMZN', 'SPX',
                                                'AMZN_OTC_1W_ATM_CALL', 
                                                'SPX_OTC_3M_ATM_CALL', 
                                                ]),
            holdings=[0.,
                      0.,  
                      -10., 
                      -10.],
            name="AMZN and SPX and AMZN_1W_Call and SPX_3M_CALL"
        )
        self._set_up_greek_bot_test(market, portfolio)

    def _set_up_greek_bot_test(self, market, portfolio, use_bs_delta=True, scenario=None):
        market.init_portfolio(portfolio)
        if scenario:
            market.load_scenario(scenario)
        else:
            market.set_pred_mode(True)
            market.set_pred_episodes(1_000)
        spec = specs.make_environment_spec(market)
        
        # Construct the agent.
        agent = GreekHedgeBot(
            portfolio=portfolio,
            use_bs_delta=use_bs_delta,
            environment_spec=spec
        )
        # Try running the environment loop. We have no assertions here because all
        # we care about is that the agent runs without raising any errors.
        loop = acme.EnvironmentLoop(market, agent)
        loop.run(num_episodes=market.get_pred_episodes())
        agent._predictor._update_progress_figures()
        status = agent._predictor._progress_measures
        print("Greek Bot PnL mean %s" % str(status['pnl_mean']))
        print("Greek Bot PnL std %s" % str(status['pnl_std']))
        print("Greek Bot 95VaR %s" % status['pnl_95VaR'])
        print("Greek Bot 99VaR %s" % status['pnl_99VaR'])
        print("Greek Bot 95CVaR %s" % status['pnl_95CVaR'])
        print("Greek Bot 99CVaR %s" % status['pnl_99CVaR'])


if __name__ == '__main__':
    # unittest.main()
    DeltaBotTest().test_regression_deltabot()
    DeltaBotTest().test_regression_deltabot()
