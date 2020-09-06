import acme
from acme import specs
from acme import  wrappers

from hb.market_env import hedging_market_env
from hb.market_env.pathgenerators import gbm_pathgenerator
from hb.market_env.rewardrules import sqrpenalty_reward
from hb.market_env.rewardrules import pnl_intrinsic_reward
from hb.market_env.rewardrules import cash_flow_reward
from hb.market_env import market_specs

import unittest


class HedgeEnvTest(unittest.TestCase):

    def test_qtable_hedge_bot(self):
        from hb.bots.qtablebot.bot import QTableBot
        # Create Environment
        gbm = gbm_pathgenerator.GBMGenerator(
            initial_price=50., drift=0.05,
            div=0.02, sigma=0.15, num_step=3, step_size=1./360.
        )
        pnl_reward = pnl_sqrpenalty_reward.PnLSquarePenaltyReward(scale_k=0.2)
        market_param = market_specs.MarketEnvParam(
            stock_ticker_size=0.1,
            stock_price_lower_bound=40.,
            stock_price_upper_bound=60.,
            lot_size=1,
            buy_sell_lots_bound=4,
            holding_lots_bound=4)
        environment = hedging_market_env.HedgingMarketEnv(
            stock_generator=gbm,
            reward_rule=pnl_reward,
            market_param=market_param,
            trading_cost_pct=0.01,
            risk_free_rate=0.,
            discount_rate=0.,
            option_maturity=30./360.,
            option_strike=50.,
            option_holding=4,
            obs_attr=['remaining_time',
                      'stock_holding',
                      'stock_price']
        )
        spec = specs.make_environment_spec(environment)
        bot = QTableBot(environment_spec=spec)
        # Try running the environment loop. We have no assertions here because all
        # we care about is that the agent runs without raising any errors.
        loop = acme.EnvironmentLoop(environment, bot)
        loop.run(num_episodes=1)

    def test_delta_hedge_bot(self):
        from hb.bots.euro_deltabot.bot import DeltaHedgeBot
        # Create Environment
        gbm = gbm_pathgenerator.GBMGenerator(
            initial_price=50., drift=0.05,
            div=0.02, sigma=0.15, num_step=3, step_size=30. / 360.,
        )
        intrinsic_reward = pnl_intrinsic_reward.PnLIntrinsicReward()
        cf_reward = cash_flow_reward.CashFlowReward()
        market_param = market_specs.MarketEnvParam(
            stock_ticker_size=1.,
            stock_price_lower_bound=45.,
            stock_price_upper_bound=55.,
            lot_size=1,
            buy_sell_lots_bound=4,
            holding_lots_bound=12)
        environment = wrappers.SinglePrecisionWrapper(hedging_market_env.HedgingMarketEnv(
            stock_generator=gbm,
            reward_rule=intrinsic_reward,
            market_param=market_param,
            trading_cost_pct=0.01,
            risk_free_rate=0.,
            discount_rate=0.,
            option_maturity=90. / 360.,
            option_strike=50.,
            option_holding=-10,
            initial_stock_holding=5
        ))
        delta_bot_env_attr = ['remaining_time', 'option_holding', 'option_strike',
                              'interest_rate', 'stock_price', 'stock_dividend',
                              'stock_sigma', 'stock_holding']
        environment.set_obs_attr(delta_bot_env_attr)
        spec = specs.make_environment_spec(environment)
        bot = DeltaHedgeBot(environment_spec=spec)
        # Try running the environment loop. We have no assertions here because all
        # we care about is that the agent runs without raising any errors.
        loop = acme.EnvironmentLoop(environment, bot)
        loop.run(num_episodes=100)

