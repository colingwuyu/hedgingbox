import acme
from acme import specs

from hb.market_env import hedging_market_env
from hb.market_env.pathgenerators import gbm_pathgenerator
from hb.market_env.rewardrules import pnl_sqrpenalty_reward
from hb.market_env import market_specs
from hb.bots.qtablebot.qtable import  QTable

import unittest


class QTableTest(unittest.TestCase):

    def test_qtable(self):
        from hb.bots.qtablebot.actor import QTableActor
        gbm = gbm_pathgenerator.GBMGenerator(
            initial_price=50., drift=0.05,
            div=0.02, sigma=0.15, num_step=3, step_size=1. / 365.
        )
        pnl_reward = pnl_sqrpenalty_reward.PnLSquarePenaltyReward(scale_k=0.2)
        market_param = market_specs.MarketEnvParam(
            stock_ticker_size=0.1,
            stock_price_lower_bound=40.,
            stock_price_upper_bound=60.,
            lot_size=1,
            buy_sell_lots_bound=4,
            holding_lots_bound=12)
        environment = hedging_market_env.HedgingMarketEnv(
            stock_generator=gbm,
            reward_rule=pnl_reward,
            market_param=market_param,
            trading_cost_pct=0.01,
            risk_free_rate=0.,
            discount_rate=0.,
            option_maturity=30. / 365.,
            option_strike=50.,
            option_holding=4,
            obs_attr=['remaining_time',
                      'stock_price',
                      'stock_holding']
        )
        spec = specs.make_environment_spec(environment)
        qtable = QTable(spec.observations, spec.actions)
        actor = QTableActor(qtable, 0.)

        num_episode = 1
        for episode in range(num_episode):
            timestep = environment.reset()

            while not timestep.last():
                action = qtable.select_maxQ_action(timestep.observation)
                timestep = environment.step(action)

