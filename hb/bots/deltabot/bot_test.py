"""Tests for the Delta Hedging bot."""

import acme
from acme import specs
from hb.bots import deltabot
from hb.market_env import bs_euro_hedge_env
import unittest
import numpy as np
import matplotlib.pyplot as plt


class DeltaBotTest(unittest.TestCase):

    def test_deltaactor(self):
        environment = bs_euro_hedge_env.BSEuroHedgeEnv(
            episode_steps=10_000,
            option_maturity=30./360.,
            option_holding=-1_000,
            stock_drift=0.0,
            max_sell_action=-1_000,
            max_buy_action=1_000,
            trading_cost_pct=0.,
        )
        actor = deltabot.DeltaHedgeActor(environment.action_spec())
        num_episode = 1
        for episode in range(num_episode):
            pnl = np.array([])
            buy_sells = np.array([])
            option_price = np.array([])
            stock_price = np.array([])
            timestep = environment.reset()

            while not timestep.last():
                action = actor.select_action(timestep.observation)
                timestep = environment.step(action)
                buy_sells = np.append(buy_sells, action)
                pnl = np.append(pnl, timestep.reward)
                option_price = np.append(option_price, timestep.observation[1])
                stock_price = np.append(stock_price, timestep.observation[6])
            plt.plot(pnl)
            print(np.mean(pnl))
            print(np.std(pnl))
            # plt.plot(buy_sells)
            # plt.plot(option_price)
            # plt.plot(stock_price)

        plt.show()

    def test_deltabot(self):
        # Create a fake environment to test with.
        environment = bs_euro_hedge_env.BSEuroHedgeEnv(
            episode_steps=30,
            option_maturity=30./360.
        )
        spec = specs.make_environment_spec(environment)

        # Construct the agent.
        agent = deltabot.DeltaHedgeBot(
            environment_spec=spec
        )

        # Try running the environment loop. We have no assertions here because all
        # we care about is that the agent runs without raising any errors.
        loop = acme.EnvironmentLoop(environment, agent)
        loop.run(num_episodes=100)


if __name__ == '__main__':
    # unittest.main()
    DeltaBotTest().test_deltaactor()
