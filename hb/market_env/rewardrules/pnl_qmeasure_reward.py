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
        pnl = super().step_reward(step_type, next_step_obs, action)
        pnl_qmeasure = - \
            abs(pnl - next_step_obs['interest_rate'] * (next_step_obs['remaining_time']-self._this_step_obs['remaining_time'])
                (self._this_step_obs['stock_holding'] * self._this_step_obs['stock_price']
                 + self._this_step_obs['option_price']*self._this_step_obs['option_holding'])) - self._scale_k*pnl**2
        return pnl_qmeasure
