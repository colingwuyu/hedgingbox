from hb.market_env.rewardrules import reward_rule
import dm_env
from acme import types
from typing import Dict


class PnLIntrinsicReward(reward_rule.RewardRule):
    def __init__(self):
        """Use intrinsic value as rough pricing model in intermediate time step
        """
        self._this_step_obs = None
        self._first_reward = True


    def step_reward(self, step_type: dm_env.StepType,
                    next_step_obs: Dict, action: types.NestedArray,
                    ) -> types.NestedArray:
        # R_i = V_{i+1} - V_i + H_i(S_{i+1} - S_i) - k|S_{i+1}*(H_{i+1}-H_i)|
        # A_i = H_{i+1} - H_i
        # V is the intrinsic value of option
        if self._first_reward:
            this_step_option_value = self._this_step_obs['option_price']
            self._first_reward = False
        else:
            this_step_option_value = self._this_step_obs['option_intrinsic_value']
        
        if abs(next_step_obs['remaining_time'] - 0) < 1e-6:
            # option expires
            # action is to liquidate all holding
            # cashflow from option_payoff
            buy_sell_action = -self._this_step_obs['stock_holding'] 
        else:
            buy_sell_action = action[0]
        
        pnl = (next_step_obs['option_intrinsic_value'] - this_step_option_value) * self._this_step_obs['option_holding'] \
            + self._this_step_obs['stock_holding'] * (next_step_obs['stock_price'] - self._this_step_obs['stock_price']) - \
            next_step_obs['stock_trading_cost_pct'] * abs(buy_sell_action) * next_step_obs['stock_price']
        self._this_step_obs = next_step_obs.copy()
        return pnl

    def reset(self, reset_obs):
        self._this_step_obs = reset_obs.copy()
        self._first_reward = True
