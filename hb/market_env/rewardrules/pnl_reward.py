from hb.market_env.rewardrules import reward_rule
import dm_env
from acme import types
from typing import Dict


class PnLReward(reward_rule.RewardRule):
    def __init__(self):
        self._this_step_obs = None

    def step_reward(self, step_type: dm_env.StepType,
                    next_step_obs: Dict, action: types.NestedArray,
                    ) -> types.NestedArray:
        buy_sell_action = action[0]
        # R_i = V_{i+1} - V_i + H_i(S_{i+1} - S_i) - k|S_{i+1}*(H_{i+1}-H_i)|
        # A_i = H_{i+1} - H_i
        pnl = (next_step_obs['option_price'] - self._this_step_obs['option_price']) * \
            self._this_step_obs['option_holding'] \
            + self._this_step_obs['stock_holding'] * \
            (next_step_obs['stock_price'] - self._this_step_obs['stock_price'])
        if next_step_obs['remaining_time'] == 0:
            # Option expires
            # add liquidation cost
            pnl -= next_step_obs['stock_trading_cost_pct'] * \
                abs(next_step_obs['stock_holding']) * \
                next_step_obs['stock_price']
        else:
            pnl -= next_step_obs['stock_trading_cost_pct'] * \
                abs(buy_sell_action)*next_step_obs['stock_price']
        self._this_step_obs = next_step_obs.copy()
        return pnl

    def reset(self, reset_obs):
        self._this_step_obs = reset_obs.copy()
