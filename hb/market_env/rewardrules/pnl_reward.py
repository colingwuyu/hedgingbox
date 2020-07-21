from hb.market_env.rewardrules import reward_rule
import dm_env
from acme import types
from typing import Dict


class PnLReward(reward_rule.RewardRule):
    def __init__(self):
        self._prev_state = None

    def step_reward(self, step_type: dm_env.StepType,
                    observation: Dict, action: types.NestedArray) -> types.NestedArray:
        buy_sell_action = action[0]
        # if step_type == dm_env.StepType.MID:
        #     if self._prev_state is None:
        #         # First reward
        #         # r_0 = -k|S_0*H_0|
        #         pnl = -observation['stock_trading_cost_pct'] * \
        #             observation['stock_price']*abs(buy_sell_action)
        #     else:
        # Mid reward
        # r_i = V_i - V_{i-1} + H_{i-1}(S_i - S_{i-1}) - k|S_i*(H_i-H_{i-1})|
        # A_i = H_i - H_{i-1}
        pnl = (observation['option_price'] - self._prev_state['option_price']) * \
            observation['option_holding'] \
            + self._prev_state['stock_holding'] * \
            (observation['stock_price'] - self._prev_state['stock_price']) \
            - observation['stock_trading_cost_pct'] * \
            observation['stock_price']*abs(buy_sell_action)
        self._prev_state = observation.copy()
        # elif step_type == dm_env.StepType.LAST:
        #     # Last reward
        #     pnl = -observation['stock_trading_cost_pct'] * \
        #         observation['stock_price'] * \
        #         observation['stock_holding']
        return pnl

    def reset(self, reset_observation):
        self._prev_state = reset_observation
