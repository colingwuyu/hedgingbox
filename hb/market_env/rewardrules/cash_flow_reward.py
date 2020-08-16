from hb.market_env.rewardrules import reward_rule
import dm_env
from acme import types
from typing import Dict


class CashFlowReward(reward_rule.RewardRule):
    def __init__(self):
        self._this_step_obs = None
        self._option_premium = 0.
        self._first_reward = True

    def step_reward(self, step_type: dm_env.StepType,
                    next_step_obs: Dict, action: types.NestedArray,
                    ) -> types.NestedArray:
        cash_flow = 0.
        if next_step_obs['remaining_time'] == 0:
            # option expires
            # action is to liquidate all holding
            # cashflow from option_payoff
            buy_sell_action = -next_step_obs['stock_holding'] 
            option_payoff = next_step_obs['stock_price'] - next_step_obs['option_strike'] if next_step_obs['stock_price'] > next_step_obs['option_strike'] else 0.
        else:
            buy_sell_action = action[0]
            option_payoff = 0.
        # stock cash flow from stock buy/sell and the transaction cost
        stock_cash_flow = -next_step_obs['stock_price']*buy_sell_action - \
                           next_step_obs['stock_trading_cost_pct'] * \
                           abs(buy_sell_action)*next_step_obs['stock_price']
        # option cash flow from payoff and premium
        option_cash_flow = option_payoff*next_step_obs['option_holding']
        if self._first_reward:
            # first step cash flow includes option premium 
            option_cash_flow += self._option_premium
            self._first_reward = False
        cash_flow = stock_cash_flow + option_cash_flow
        return cash_flow

    def reset(self, reset_obs):
        self._this_step_obs = reset_obs.copy()
        self._option_premium = self._this_step_obs['option_price']
        self._first_reward = True
        
