from hb.market_env.rewardrules import pnl_reward
import dm_env
from acme import types
from typing import Dict


class PnLQMeasureReward(pnl_reward.PnLReward):
    def __init__(self, scale_k: float = 0.5):
        """Constructor of PnL with Squared PnL Penalty Reward
           $Reward = -|PnL_t-r_{t-1}*d_t(P_{t-1})| - k*(PnL_t)^2$
        here $P_{t-1}$ is the portfolio value $H_{t-1}S_{t-1}+V_{t-1}$
        Args:
            scale_k (float): scaling factor for squared PnL Penalty term
        """
        self._scale_k = scale_k
        super().__init__()

    def step_reward(self, step_type: dm_env.StepType,
                    next_step_obs: Dict, action: types.NestedArray) -> types.NestedArray:
        buy_sell_action = action[0]
        # PnL
        pnl = (next_step_obs['option_price'] - self._this_step_obs['option_price']) * \
            next_step_obs['option_holding'] \
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
        pnl_qmeasure = - abs(pnl -
                             next_step_obs['interest_rate'] *
                             (self._this_step_obs['remaining_time']-next_step_obs['remaining_time']) *
                             (self._this_step_obs['stock_holding'] * self._this_step_obs['stock_price']
                                 + self._this_step_obs['option_holding']*self._this_step_obs['option_price'])) \
            - self._scale_k*pnl**2
        self._this_step_obs = next_step_obs.copy()
        return pnl_qmeasure
