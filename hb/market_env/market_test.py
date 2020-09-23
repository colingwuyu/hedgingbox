import unittest
from hb.transaction_cost.percentage_transaction_cost import PercentageTransactionCost
from hb.instrument.instrument_factory import InstrumentFactory
from hb.market_env.rewardrules.pnl_reward import PnLReward
from hb.market_env.market import Market
from hb.market_env.portfolio import Portfolio
from hb.utils.process import *

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
            'Stock AMZN 100 10 0 0.'
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
                        risk_free_rate=0.0223,
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
            'Stock SPX 100 10 1.92 0.5'
        )
        # maturity = ['1W', '2W', '3W', '4W', '7W', '2M', '3M']
        # strike = [3330, 3335, 3340, 3345, 3350, 3355]
        # iv = [  
        #     [22.02,21.96,21.90,23.03,22.51,22.95],
        #     [22.14,23.02,21.92,22.26,21.71,22.59],
        #     [22.26,22.17,22.03,22.32,21.81,21.74],
        #     [23.01,22.27,22.63,21.04,22.36,21.59],
        #     [22.29,21.46,21.34,21.24,21.46,21.01],
        #     [24.12,22.16,22.07,21.95,22.59,22.17],
        #     [24.06,23.96,23.88,23.79,23.70,23.61]
        # ]
        # spx_listed_calls = []
        # n = 0
        # for i, m in enumerate(maturity):
        #     spx_listed_calls_m = []
        #     for j, s in enumerate(strike):
        #         spx_listed_calls_m = spx_listed_calls_m \
        #                 + [InstrumentFactory.create(
        #                     f'EuroOpt SPX Listed {m} Call {s} {iv[i][j]} 0.5 (SPX_Call{n})'
        #                 ).underlying(spx)] 
        #         n += 1
        #     spx_listed_calls.append(spx_listed_calls_m)
        
        # Heston param from 
        heston_param = HestonProcessParam(
            risk_free_rate = 0.0223, spot = spx.get_quote(), spot_var = 0.001006,
            drift = spx.get_annual_yield(), dividend = spx.get_dividend_yield(),
            kappa = 2.4056, theta = 0.04264, vov = 0.8121, rho = -0.7588,
            use_risk_free = False
        )
        # heston_param = HestonProcessParam(
        #     risk_free_rate=0.015,
        #     spot=spx.get_quote(), 
        #     drift=spx.get_annual_yield(), 
        #     dividend=spx.get_dividend_yield(),
        #     spot_var=0.096024, kappa=6.288453, theta=0.397888, 
        #     rho=-0.696611, vov=0.753137, use_risk_free=False
        # )
        market.calibrate(underlying=spx,
                         param=heston_param)
        # List of OTM options for variance swap replication
        k0 = 100
        call_strikes = range(k0, 150, 5)
        put_strikes = range(k0, 50, -5)
        replicating_opts = []
        for i, strike in enumerate(put_strikes):
            # OTM put
            otm_put = InstrumentFactory.create(
                            f'EuroOpt SPX Listed 3M Put {strike} 25 0. (SPX_Listed_3M_PUT{i})'
                        ).underlying(spx)
            market.add_instruments([otm_put])
            replicating_opts += [f'SPX_Listed_3M_PUT{i}']
        for i, strike in enumerate(call_strikes):
            # OTM call
            otm_call = InstrumentFactory.create(
                            f'EuroOpt SPX Listed 3M Call {strike} 25 0. (SPX_Listed_3M_CALL{i})'
                        ).underlying(spx)
            market.add_instruments([otm_call])
            replicating_opts += [f'SPX_Listed_3M_CALL{i}']
        variance_swap_opt_hedging = market.get_instruments(replicating_opts)
                                                            
        variance_swap = InstrumentFactory.create(
            f'VarSwap SPX 3M 10.65 0.1 (SPX_3M_VAR_SWAP)'
        ).underlying(spx).replicating_opts(variance_swap_opt_hedging)
        variance_swap.set_pricing_method('Heston')
        # variance_swap.set_excl_realized_var(True)
        market.add_instruments([variance_swap])
        # Test Var Swap
        # test = InstrumentFactory.create(
        #     'Stock TEST 100 5 0 0.5'
        # )
        # maturity = ['3M']
        # strike = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135]
        # iv = [  
        #     [30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13],
        # ]
        # test_listed_calls = []
        # n = 0
        # for i, m in enumerate(maturity):
        #     test_listed_calls_m = []
        #     for j, s in enumerate(strike):
        #         test_listed_calls_m = test_listed_calls_m \
        #                 + [InstrumentFactory.create(
        #                     f'EuroOpt TEST Listed {m} Call {s} {iv[i][j]} 0.5 (TEST_Call{n})'
        #                 ).underlying(test)] 
        #         n += 1
        #     test_listed_calls.append(test_listed_calls_m)
        
        # market.calibrate(underlying=test,
        #                  listed_options=test_listed_calls)
        
        # listed_3m_put_1 = InstrumentFactory.create(
        #                         f'EuroOpt TEST Listed 3M Put 50 30 0.5 (TEST_Listed_3M_PUT1)'
        #                     ).underlying(test)
        # listed_3m_put_2 = InstrumentFactory.create(
        #                         f'EuroOpt TEST Listed 3M Put 55 29 0.5 (TEST_Listed_3M_PUT2)'
        #                     ).underlying(test)
        # listed_3m_put_3 = InstrumentFactory.create(
        #                         f'EuroOpt TEST Listed 3M Put 60 28 0.5 (TEST_Listed_3M_PUT3)'
        #                     ).underlying(test)
        # listed_3m_put_4 = InstrumentFactory.create(
        #                         f'EuroOpt TEST Listed 3M Put 65 27 0.5 (TEST_Listed_3M_PUT4)'
        #                     ).underlying(test)
        # listed_3m_put_5 = InstrumentFactory.create(
        #                         f'EuroOpt TEST Listed 3M Put 70 26 0.5 (TEST_Listed_3M_PUT5)'
        #                     ).underlying(test)
        # listed_3m_put_6 = InstrumentFactory.create(
        #                         f'EuroOpt TEST Listed 3M Put 75 25 0.5 (TEST_Listed_3M_PUT6)'
        #                     ).underlying(test)
        # listed_3m_put_7 = InstrumentFactory.create(
        #                         f'EuroOpt TEST Listed 3M Put 80 24 0.5 (TEST_Listed_3M_PUT7)'
        #                     ).underlying(test)
        # listed_3m_put_8 = InstrumentFactory.create(
        #                         f'EuroOpt TEST Listed 3M Put 85 23 0.5 (TEST_Listed_3M_PUT8)'
        #                     ).underlying(test)
        # listed_3m_put_9 = InstrumentFactory.create(
        #                         f'EuroOpt TEST Listed 3M Put 90 22 0.5 (TEST_Listed_3M_PUT9)'
        #                     ).underlying(test)
        # listed_3m_put_10 = InstrumentFactory.create(
        #                         f'EuroOpt TEST Listed 3M Put 95 21 0.5 (TEST_Listed_3M_PUT10)'
        #                     ).underlying(test)
        # listed_3m_put_11 = InstrumentFactory.create(
        #                         f'EuroOpt TEST Listed 3M Put 100 20 0.5 (TEST_Listed_3M_PUT11)'
        #                     ).underlying(test)
        # market.add_instruments([listed_3m_put_1, listed_3m_put_2, listed_3m_put_3, listed_3m_put_4,
        #                         listed_3m_put_5, listed_3m_put_6, listed_3m_put_7, listed_3m_put_8,
        #                         listed_3m_put_9, listed_3m_put_10, listed_3m_put_11])
        # variance_swap_opt_hedging = market.get_instruments(['TEST_Listed_3M_PUT1',
        #                                                     'TEST_Listed_3M_PUT2',
        #                                                     'TEST_Listed_3M_PUT3',
        #                                                     'TEST_Listed_3M_PUT4',
        #                                                     'TEST_Listed_3M_PUT5',
        #                                                     'TEST_Listed_3M_PUT6',
        #                                                     'TEST_Listed_3M_PUT7',
        #                                                     'TEST_Listed_3M_PUT8',
        #                                                     'TEST_Listed_3M_PUT9',
        #                                                     'TEST_Listed_3M_PUT10',
        #                                                     'TEST_Listed_3M_PUT11',
        #                                                     'TEST_Call10',
        #                                                     'TEST_Call11',
        #                                                     'TEST_Call12',
        #                                                     'TEST_Call13',
        #                                                     'TEST_Call14',
        #                                                     'TEST_Call15',
        #                                                     'TEST_Call16',
        #                                                     'TEST_Call17',
        #                                                     ])
                                                            
        # variance_swap = InstrumentFactory.create(
        #     f'VarSwap TEST 3M 25 1 50000 (TEST_3M_VAR_SWAP)'
        # ).underlying(test).replicating_opts(variance_swap_opt_hedging)
        # market.add_instruments([variance_swap])
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
    