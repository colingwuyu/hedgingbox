import acme
from acme import specs
from acme import wrappers
from acme import environment_loop
import numpy as np

from hb.market_env import hedging_market_env
from hb.market_env.pathgenerators import gbm_pathgenerator
from hb.market_env import market_specs
from hb.market_env.rewardrules.pnl_reward import PnLReward
from hb.market_env.rewardrules.pnl_sqrpenalty_reward import PnLSquarePenaltyReward
from hb.bots.qtablebot.qtable import  QTable
from hb.bots.deltabot.bot import DeltaHedgeBot
import matplotlib.pyplot as plt

import dm_env
import unittest


class QTableTest(unittest.TestCase):

    def test_qtable(self):
        from hb.bots.qtablebot.actor import QTableActor
        trading_cost_pct = 0. #@param {type:"number"}
        gbm = gbm_pathgenerator.GBMGenerator(
                    initial_price=50., drift=0.05,
                    div=0.0, sigma=0.15, num_step=3, step_size=30./360.,
                )
        pnl_penalty_reward = PnLSquarePenaltyReward(scale_k=1e-3)
        pnl_reward = PnLReward()
        market_param = market_specs.MarketEnvParam(
            stock_ticker_size=1.,
            stock_price_lower_bound=45.,
            stock_price_upper_bound=55.,
            lot_size=1,
            buy_sell_lots_bound=4,
            holding_lots_bound=17)
        environment = wrappers.SinglePrecisionWrapper(hedging_market_env.HedgingMarketEnv(
                    stock_generator=gbm,
                    reward_rule=pnl_penalty_reward,
                    market_param=market_param,
                    trading_cost_pct=trading_cost_pct,
                    risk_free_rate=0.,
                    discount_rate=0.,
                    option_maturity=450./360.,
                    option_strike=50.,
                    option_holding=-10,
                    initial_stock_holding=5
                ))
        environment.set_repeat_path(1)
        qtable_bot_env_attr = ['remaining_time', 'stock_holding', 'stock_price']
        environment.set_obs_attr(qtable_bot_env_attr)

        import pickle
        with open("zerotradingcost_qtable.pickle", 'rb') as pf:
            qtable = pickle.load(pf)            
        pred_num_episodes = 1000 #@param {type:"integer"}
        actor = QTableActor(qtable, epsilon=0.)
        qtable_action_list = np.array([])
        qtable_pnl_list = np.array([])
        qtable_reward_list = np.array([])
        figure = plt.figure()
        for episode in range(pred_num_episodes):
            episode_pnl = 0.
            episode_reward = 0.
            episode_pnl_path = np.array([])
            episode_action = np.array([])
            timestep = environment.reset()
            pnl_reward.reset(environment._state_values)

            while not timestep.last():
                action = actor.select_action(timestep.observation)
                episode_action = np.append(episode_action, action)
                timestep = environment.step(action)
                pnl = pnl_reward.step_reward(
                        dm_env.StepType.MID,
                        environment._state_values,
                        action)
                episode_pnl += pnl
                episode_reward += timestep.reward
                episode_pnl_path = np.append(episode_pnl_path, pnl)
            qtable_pnl_list = np.append(qtable_pnl_list, episode_pnl)
            qtable_reward_list =  np.append(qtable_reward_list, episode_reward)
            qtable_action_list = np.append(qtable_action_list, episode_action)

        delta_bot_env_attr = ['remaining_time', 'option_holding', 'option_strike',
                      'interest_rate', 'stock_price', 'stock_dividend',
                      'stock_sigma', 'stock_holding']
        environment.set_obs_attr(delta_bot_env_attr)
        spec = specs.make_environment_spec(environment)
        delta_bot = DeltaHedgeBot(environment_spec=spec)

        #@title Predict by Delta Hedging Bot
        delta_action_list = np.array([])
        delta_pnl_list = np.array([])
        delta_reward_list = np.array([])
        figure = plt.figure()
        for episode in range(pred_num_episodes):
            episode_pnl = 0.
            episode_reward = 0.
            episode_pnl_path = np.array([])
            episode_action = np.array([])
            timestep = environment.reset()
            pnl_reward.reset(environment._state_values)
            
            while not timestep.last():
                action = delta_bot.select_action(timestep.observation)
                episode_action = np.append(episode_action, action)
                timestep = environment.step(action)
                pnl = pnl_reward.step_reward(
                        dm_env.StepType.MID,
                        environment._state_values,
                        action)
                episode_pnl += pnl
                episode_reward += timestep.reward
                episode_pnl_path = np.append(episode_pnl_path, pnl)
            delta_pnl_list = np.append(delta_pnl_list, episode_pnl)
            delta_reward_list = np.append(delta_reward_list, episode_reward)
            delta_action_list = np.append(delta_action_list, episode_action)

        plt.hist(qtable_pnl_list, alpha=0.5, label='QTable')
        plt.hist(delta_pnl_list, alpha=0.5, label='Delta')
        plt.title('Total PnL Distribution')
        plt.legend(loc='upper right')
        plt.savefig('zerotradingcost_cost.png')
        plt.show()

        plt.hist(qtable_reward_list, alpha=0.5, label='QTable')
        plt.hist(delta_reward_list, alpha=0.5, label='Delta')
        plt.title('Total Reward Distribution')
        plt.legend(loc='upper right')
        plt.show()

    def test_qtable_predictor(self):
        from hb.bots.qtablebot.predictor import QTablePredictor
        from hb.bots.qtablebot.actor import QTableActor
        trading_cost_pct = 0. #@param {type:"number"}
        gbm = gbm_pathgenerator.GBMGenerator(
                    initial_price=50., drift=0.05,
                    div=0.0, sigma=0.15, num_step=3, step_size=30./360.,
                )
        pnl_penalty_reward = PnLSquarePenaltyReward(scale_k=1e-3)
        market_param = market_specs.MarketEnvParam(
            stock_ticker_size=1.,
            stock_price_lower_bound=45.,
            stock_price_upper_bound=55.,
            lot_size=1,
            buy_sell_lots_bound=4,
            holding_lots_bound=17)
        environment = wrappers.SinglePrecisionWrapper(hedging_market_env.HedgingMarketEnv(
                    stock_generator=gbm,
                    reward_rule=pnl_penalty_reward,
                    market_param=market_param,
                    trading_cost_pct=trading_cost_pct,
                    risk_free_rate=0.,
                    discount_rate=0.,
                    option_maturity=450./360.,
                    option_strike=50.,
                    option_holding=-10,
                    initial_stock_holding=5
                ))
        qtable_bot_env_attr = ['remaining_time', 'stock_holding', 'stock_price']
        environment.set_obs_attr(qtable_bot_env_attr)

        import pickle
        with open("zerotradingcost_qtable.pickle", 'rb') as pf:
            qtable = pickle.load(pf)            
        pred_num_episodes = 1000 #@param {type:"integer"}
        actor = QTableActor(qtable, epsilon=0.8)
        predictor = QTablePredictor(actor)
        
        for episode in range(pred_num_episodes):
            timestep = environment.reset()
            predictor.observe_first(timestep)
            while not timestep.last():
                action = predictor.select_action(timestep.observation)
                timestep = environment.step(action)
                predictor.observe(action, timestep)
        predictor.log_pred_perf()
                
        delta_bot_env_attr = ['remaining_time', 'option_holding', 'option_strike',
                      'interest_rate', 'stock_price', 'stock_dividend',
                      'stock_sigma', 'stock_holding']
        environment.set_obs_attr(delta_bot_env_attr)
        spec = specs.make_environment_spec(environment)
        delta_bot = DeltaHedgeBot(environment_spec=spec, 
                                  pred_episode=pred_num_episodes)

        loop = acme.EnvironmentLoop(environment, delta_bot)
        loop.run(num_episodes=pred_num_episodes)