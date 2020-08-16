from hb.market_env.rewardrules import reward_rule
import dm_env
from acme import types
from typing import Dict


class PnLReward(reward_rule.RewardRule):
    def __init__(self):
        self._this_step_obs = None
        self._first_reward = True

    def step_reward(self, step_type: dm_env.StepType,
                    next_step_obs: Dict, action: types.NestedArray,
                    ) -> types.NestedArray:
        # R_i = V_{i+1} - V_i + H_i(S_{i+1} - S_i) - k|S_{i+1}*(H_{i+1}-H_i)|
        # A_i = H_{i+1} - H_i
        if abs(next_step_obs['remaining_time'] - 0) < 1e-6:
            # option expires
            # action is to liquidate all holding
            # cashflow from option_payoff
            buy_sell_action = -self._this_step_obs['stock_holding'] 
        else:
            buy_sell_action = action[0]
        
        pnl = (next_step_obs['option_price'] - self._this_step_obs['option_price']) * self._this_step_obs['option_holding'] \
            + self._this_step_obs['stock_holding'] * (next_step_obs['stock_price'] - self._this_step_obs['stock_price']) - \
            next_step_obs['stock_trading_cost_pct'] * abs(buy_sell_action) * next_step_obs['stock_price']
        self._this_step_obs = next_step_obs.copy()
        return pnl

    def reset(self, reset_obs):
        self._this_step_obs = reset_obs.copy()
        self._first_reward = True


class FwdPnLReward(reward_rule.RewardRule):
    def __init__(self):
        """Forward PnL Reward
        """
        self._this_step_obs = None
        self._first_reward = True

    def step_reward(self, step_type: dm_env.StepType,
                    next_step_obs: Dict, action: types.NestedArray,
                    ) -> types.NestedArray:
        if next_step_obs['remaining_time'] < -1e-6:
            # pass option expiry
            return 0.
        # hedging buy/sell action happens at time i
        buy_sell_action = action[0]
        if abs(next_step_obs['remaining_time'] - 0) < 1e-6:
            # next_step is terminal step when option expires
            # include the cost to liquidate all hedging positions
            liquidation_transac_cost = next_step_obs['stock_trading_cost_pct'] * abs(next_step_obs['stock_holding']) * next_step_obs['stock_price']
        else:
            liquidation_transac_cost = 0.

        # option pnl from time i to time i+1
        option_pnl = (next_step_obs['option_price'] - self._this_step_obs['option_price']) * next_step_obs['option_holding']
        # stock pnl from time i to time i+1
        stock_pnl = (next_step_obs['stock_price'] - self._this_step_obs['stock_price']) * next_step_obs['stock_holding']  
        # transaction cost happens at time i.
        transac_cost = self._this_step_obs['stock_trading_cost_pct'] * abs(buy_sell_action) * self._this_step_obs['stock_price'] + liquidation_transac_cost
        
        pnl = option_pnl + stock_pnl - transac_cost 
        self._this_step_obs = next_step_obs.copy()
        return pnl

    def reset(self, reset_obs):
        self._this_step_obs = reset_obs.copy()
        self._first_reward = True
