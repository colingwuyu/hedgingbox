import unittest
from hb.transaction_cost.percentage_transaction_cost import PercentageTransactionCost
from hb.instrument.instrument_factory import InstrumentFactory
from hb.market_env.rewardrules.pnl_reward import PnLReward
from hb.market_env.market import Market
from hb.market_env.portfolio import Portfolio

class MarketTest(unittest.TestCase):

    def set_up_bsm_market(self):
        market = Market(reward_rule=PnLReward(),risk_free_rate=0.015,hedging_step_in_days=1)
        # create instruments
        amzn = InstrumentFactory.create(
            'Stock AMZN 3400 25 0 0.15'
        )
        otc_atm_3m_call = InstrumentFactory.create(
                                f'EuroOpt AMZN OTC 3M Call 3400 49.25 5 (AMZN_OTC_3M_ATM_CALL)'
                            ).underlying(amzn)
        market.calibrate(vol_model='BSM',
                         underlying=amzn,
                         listed_options=otc_atm_3m_call)
        portfolio = Portfolio(
            instruments=market.get_instruments(['AMZN', 'AMZN_OTC_3M_ATM_CALL']),
            holdings=[0., -10.]
        )
        market.init_portfolio(portfolio)
        return market

    def test_bsm_market_setup(self):
        import numpy as np
        market = self.set_up_bsm_market()
        print(market)
        pred_num_episodes = 1
        for episode in range(pred_num_episodes):
            timestep = market.reset()
            
            while not timestep.last():
                action = np.random.uniform(-4, 4, 1)
                timestep = market.step(action)
                
    def set_up_heston_market(self):
        market = Market(reward_rule=PnLReward(),risk_free_rate=0.015,hedging_step_in_days=1)
        # create instruments
        amzn = InstrumentFactory.create(
            'Stock AMZN 3400 25 0 0.15'
        )
        maturity = ['1W', '2W', '3W', '4W', '7W', '3M']
        strike = [3395, 3400, 3405, 3410]
        iv = [[33.19, 33.21, 33.08, 33.34],
            [36.08, 36.02, 36.11, 35.76],
            [38.14, 38.08, 37.89, 37.99],
            [39.99, 39.84, 40.01, 39.79],
            [43.51, 43.7, 43.67, 43.63],
            [49.34, 49.25, 48.77, 48.56]]
        amzn_listed_calls = []
        n = 0
        for i, m in enumerate(maturity):
            amzn_listed_calls_m = []
            for j, s in enumerate(strike):
                amzn_listed_calls_m = amzn_listed_calls_m \
                        + [InstrumentFactory.create(
                            f'EuroOpt AMZN Listed {m} Call {s} {iv[i][j]} 5 (AMZN_Call{n})'
                        ).underlying(amzn)] 
                n += 1
            amzn_listed_calls.append(amzn_listed_calls_m)
        
        market.calibrate(vol_model='Heston',
                        underlying=amzn,
                        listed_options=amzn_listed_calls)
        
        otc_atm_3m_call = InstrumentFactory.create(
                                f'EuroOpt AMZN OTC 3M Call 3400 49.25 5 (AMZN_OTC_3M_ATM_CALL)'
                            ).underlying(amzn)
        portfolio = Portfolio(
            instruments=market.get_instruments(['AMZN', 'AMZN_OTC_3M_ATM_CALL']),
            holdings=[0., -10.]
        )
        market.init_portfolio(portfolio)
        return market
        
    def test_heston_market_setup(self):
        import numpy as np
        market = self.set_up_heston_market()
        print(market)
        pred_num_episodes = 1
        for episode in range(pred_num_episodes):
            timestep = market.reset()
            
            while not timestep.last():
                action = np.random.uniform(-4, 4, 1)
                timestep = market.step(action)
