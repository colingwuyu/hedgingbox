import unittest
from hb.transaction_cost.percentage_transaction_cost import PercentageTransactionCost
from hb.instrument.instrument_factory import InstrumentFactory
from hb.market_env.rewardrules.pnl_reward import PnLReward
from hb.market_env.market import Market
from hb.market_env.portfolio import Portfolio

class MarketTest(unittest.TestCase):
    def set_up_regression_bsm_market(self):
        market = Market(reward_rule=PnLReward(),
                        risk_free_rate=0.02,
                        hedging_step_in_days=1,
                        vol_model='BSM',
                        name='Regression_BSM',
                        dir_='Markets')
        # create instruments
        # --------------------------------------------
        # AMZN
        # --------------------------------------------
        amzn = InstrumentFactory.create(
            'Stock AMZN 100 10 0 0.0'
        )
        otc_atm_1w_call = InstrumentFactory.create(
                                f'EuroOpt AMZN OTC 1W Call 100.10 30.0 5 (AMZN_OTC_1W_ATM_CALL)'
                            ).underlying(amzn)
        otc_atm_1m_call = InstrumentFactory.create(
                                f'EuroOpt AMZN OTC 4W Call 100.30 30.0 5 (AMZN_OTC_1M_ATM_CALL)'
                            ).underlying(amzn)
        otc_atm_3m_call = InstrumentFactory.create(
                                f'EuroOpt AMZN OTC 3M Call 100.50 30.0 5 (AMZN_OTC_3M_ATM_CALL)'
                            ).underlying(amzn)
        market.calibrate(underlying=amzn,
                         listed_options=otc_atm_3m_call)
        market.add_instruments([otc_atm_1w_call, otc_atm_1m_call, otc_atm_3m_call])
        return market

    def set_up_bsm_market(self):
        market = Market(reward_rule=PnLReward(),
                        risk_free_rate=0.015,
                        hedging_step_in_days=1,
                        vol_model='BSM',
                        name='BSM_AMZN_SPX',
                        dir_='Markets')
        # create instruments
        # --------------------------------------------
        # AMZN
        # --------------------------------------------
        amzn = InstrumentFactory.create(
            'Stock AMZN 3400 25 0 0.15'
        )
        otc_atm_3m_call = InstrumentFactory.create(
                                f'EuroOpt AMZN OTC 1W Call 3400 33.21 5 (AMZN_OTC_1W_ATM_CALL)'
                            ).underlying(amzn)
        market.calibrate(underlying=amzn,
                         listed_options=otc_atm_3m_call)
        market.add_instruments([otc_atm_3m_call])    
        # --------------------------------------------
        # SPX
        # --------------------------------------------
        spx = InstrumentFactory.create(
            'Stock SPX 3426.96 10 1.92 0.5'
        )
        otc_atm_3m_call = InstrumentFactory.create(
                                f'EuroOpt SPX OTC 3M Call 3425 27.87 3.5 (SPX_OTC_3M_ATM_CALL)'
                            ).underlying(spx)
        market.calibrate(underlying=spx,
                         listed_options=otc_atm_3m_call)
        market.add_instruments([otc_atm_3m_call])            
        return market

    def test_bsm_market_setup(self):
        import numpy as np
        from hb.utils.date import get_cur_days
        market = self.set_up_bsm_market()
        portfolio = Portfolio.make_portfolio(
            instruments=market.get_instruments([
                                                'SPX', 
                                                'SPX_OTC_3M_ATM_CALL',  
                                                ]),
            holdings=[0., 
                      -10.],
            name="SPX and SPX_3M_CALL"
        )
        market.init_portfolio(portfolio)
        print(market)
        market.set_pred_mode(True)
        market.set_pred_episodes(1_000)
        pred_num_episodes = market.get_pred_episodes()
        gbm_stock_prices = np.zeros([pred_num_episodes, 91])
        gbm_option_prices = np.zeros([pred_num_episodes, 91])
        times = np.zeros([91])
        i = 0; j=0
        for episode in range(pred_num_episodes):
            timestep = market.reset()
            j = 0
            times[j] = get_cur_days()
            gbm_stock_prices[i][j] = timestep.observation[0]
            gbm_option_prices[i][j] = timestep.observation[2]
            j += 1
            while not timestep.last():
                action = np.random.uniform(-4, 4, 1)
                timestep = market.step(action)
                times[j] = get_cur_days()
                gbm_stock_prices[i][j] = timestep.observation[0]
                gbm_option_prices[i][j] = timestep.observation[2]
                j += 1
            i += 1
        self.plot_results(pred_num_episodes, times, gbm_stock_prices, gbm_option_prices)

    def test_bsm_market_load_scenario(self):
        import numpy as np
        from hb.utils.date import get_cur_days
        market = self.set_up_bsm_market()
        portfolio = Portfolio.make_portfolio(
            instruments=market.get_instruments([
                                                'SPX', 
                                                'SPX_OTC_3M_ATM_CALL',  
                                                ]),
            holdings=[0., 
                      -10.],
            name="SPX and SPX_3M_CALL"
        )
        market.init_portfolio(portfolio)
        print(market)
        market.load_scenario('Covid19')
        pred_num_episodes = market.get_pred_episodes()
        num_steps = market.get_num_steps() + 1
        gbm_stock_prices = np.zeros([pred_num_episodes, num_steps])
        gbm_option_prices = np.zeros([pred_num_episodes, num_steps])
        times = np.zeros([num_steps])
        i = 0; j=0
        for episode in range(pred_num_episodes):
            timestep = market.reset()
            j = 0
            times[j] = get_cur_days()
            gbm_stock_prices[i][j] = timestep.observation[0]
            gbm_option_prices[i][j] = timestep.observation[2]
            j += 1
            while not timestep.last():
                action = np.random.uniform(-4, 4, 1)
                timestep = market.step(action)
                times[j] = get_cur_days()
                gbm_stock_prices[i][j] = timestep.observation[0]
                gbm_option_prices[i][j] = timestep.observation[2]
                j += 1
            i += 1
        self.plot_results(pred_num_episodes, times, gbm_stock_prices, gbm_option_prices)

    def set_up_heston_market(self):
        market = Market(reward_rule=PnLReward(),
                        risk_free_rate=0.015,
                        hedging_step_in_days=1,
                        vol_model='Heston',
                        name='Heston_AMZN_SPX',
                        dir_='Markets')
        # create instruments
        # --------------------------------------------
        # AMZN
        # --------------------------------------------
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
        
        market.calibrate(underlying=amzn,
                         listed_options=amzn_listed_calls)
        
        otc_atm_1w_call = InstrumentFactory.create(
                                f'EuroOpt AMZN OTC 1W Call 3400 33.21 5 (AMZN_OTC_1W_ATM_CALL)'
                            ).underlying(amzn)
        otc_atm_1m_call = InstrumentFactory.create(
                                f'EuroOpt AMZN OTC 4W Call 3400 39.84 5 (AMZN_OTC_1M_ATM_CALL)'
                            ).underlying(amzn)
        otc_atm_3m_call = InstrumentFactory.create(
                                f'EuroOpt AMZN OTC 3M Call 3400 49.25 5 (AMZN_OTC_3M_ATM_CALL)'
                            ).underlying(amzn)
        market.add_instruments([otc_atm_1w_call, otc_atm_1m_call, otc_atm_3m_call])
        # --------------------------------------------
        # SPX
        # --------------------------------------------
        spx = InstrumentFactory.create(
            'Stock SPX 3426.96 10 1.92 0.5'
        )
        maturity = ['1W', '2W', '3W', '4W', '7W', '3M']
        strike = [3420, 3425, 3430, 3435]
        iv = [[29.33, 28.99, 28.66, 28.32],
            [27.42, 27.78, 27.15, 27.54],
            [26.28, 26.05, 25.85, 25.62],
            [25.96, 25.77, 25.57, 25.38],
            [25.5, 25.35, 25.2, 24.93],
            [27.87, 27.75, 27.63, 27.51]]
        spx_listed_calls = []
        n = 0
        for i, m in enumerate(maturity):
            spx_listed_calls_m = []
            for j, s in enumerate(strike):
                spx_listed_calls_m = spx_listed_calls_m \
                        + [InstrumentFactory.create(
                            f'EuroOpt SPX Listed {m} Call {s} {iv[i][j]} 3.5 (SPX_Call{n})'
                        ).underlying(spx)] 
                n += 1
            spx_listed_calls.append(spx_listed_calls_m)
        
        market.calibrate(underlying=spx,
                         listed_options=spx_listed_calls)
        
        otc_atm_1w_call = InstrumentFactory.create(
                                f'EuroOpt SPX OTC 1W Call 3425 28.99 3.5 (SPX_OTC_1W_ATM_CALL)'
                            ).underlying(spx)
        otc_atm_1m_call = InstrumentFactory.create(
                                f'EuroOpt SPX OTC 4W Call 3425 25.96 3.5 (SPX_OTC_1M_ATM_CALL)'
                            ).underlying(spx)
        otc_atm_3m_call = InstrumentFactory.create(
                                f'EuroOpt SPX OTC 3M Call 3425 27.87 3.5 (SPX_OTC_3M_ATM_CALL)'
                            ).underlying(spx)
        market.add_instruments([otc_atm_1w_call, otc_atm_1m_call, otc_atm_3m_call])
        return market
        
    def test_heston_market_setup(self):
        import numpy as np
        from hb.utils.date import get_cur_days
        market = self.set_up_heston_market()
        portfolio = Portfolio.make_portfolio(
            instruments=market.get_instruments([
                                                'AMZN', 
                                                'AMZN_OTC_1W_ATM_CALL',  
                                                ]),
            holdings=[0., 
                      -10.],
            name="AMZN and AMZN_1W_CALL"
        )
        market.init_portfolio(portfolio)
        # market.set_pred_mode(True)
        market.set_pred_episodes(1_000)
        print(market)
        pred_num_episodes = market.get_pred_episodes()
        num_steps = market.get_num_steps() + 1
        heston_stock_prices = np.zeros([pred_num_episodes, num_steps])
        heston_option_prices = np.zeros([pred_num_episodes, num_steps])
        times = np.zeros([num_steps])
        i = 0; j=0
        for episode in range(pred_num_episodes):
            timestep = market.reset()
            j = 0
            times[j] = get_cur_days()
            heston_stock_prices[i][j] = timestep.observation[0]
            heston_option_prices[i][j] = timestep.observation[2]
            j += 1
            while not timestep.last():
                # print(j)
                action = np.random.uniform(-4, 4, 1)
                timestep = market.step(action)
                times[j] = get_cur_days()
                heston_stock_prices[i][j] = timestep.observation[0]
                heston_option_prices[i][j] = timestep.observation[2]
                j += 1
            i += 1
        self.plot_results(pred_num_episodes, times, heston_stock_prices, heston_option_prices)
        
    def plot_results(self, pred_num_episodes, times, prices, option_prices):
        import matplotlib.pyplot as plt

        for i in range(pred_num_episodes):
            plt.plot(times, prices[i, :], lw=0.8, alpha=0.6)
        plt.title("Simulation Spot")
        plt.show()
        print(prices[:,-1].mean())
        for i in range(pred_num_episodes):
            plt.plot(times, option_prices[i, :], lw=0.8, alpha=0.6)
        plt.title("Simulation Option")
        plt.show()
        print(option_prices[:,-1].mean())


if __name__ == "__main__":
    # MarketTest().test_bsm_market_setup()
    MarketTest().test_heston_market_setup()
    # MarketTest().test_bsm_market_load_scenario()