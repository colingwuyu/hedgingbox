from hb.market_env.rewardrules.pnl_reward import *
import dm_env
import unittest


class RewardRulesTest(unittest.TestCase):

    def test_noisy_pnl(self):
        noisy_pnl = NoisyPnLReward(noise_mean_percentage=0.0, noise_std_percentage=0.2)
        obs = {
            'remaining_time': 1.0,
            'stock_holding': 5.0,
            'stock_price': 100.,
            'option_holding': -10.,
            'option_price': 3.0,
            'stock_trading_cost_pct': 0.01
        }
        action = [0]
        noisy_pnl.reset(obs)
        noisy_pnl.step_reward(dm_env.StepType.MID, obs, action)

if __name__ == "__main__":
    a = RewardRulesTest()
    a.test_noisy_pnl()