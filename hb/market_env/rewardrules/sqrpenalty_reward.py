from hb.market_env.rewardrules import reward_rule
import dm_env
from acme import types
from typing import Dict


class SquarePenaltyReward(reward_rule.RewardRule):
    def __init__(self, reward_rule: reward_rule.RewardRule, scale_k: float = 1e-4):
        """Constructor of Reward with Squared Penalty Reward
           $Wrapped_Reward = Reward - k/2*(Reward)^2$

        Args:
            scale_k (float): scaling factor for squared reward Penalty term
        """
        self._scale_k = scale_k
        self._reward_rule = reward_rule
        super().__init__()

    def step_reward(self, step_type: dm_env.StepType,
                    next_step_obs: Dict, action: types.NestedArray,
                    extra: dict=dict()) -> types.NestedArray:
        reward = self._reward_rule.step_reward(step_type, next_step_obs, action, extra)
        reward_square_penalty = reward - 0.5*self._scale_k*reward**2
        return reward_square_penalty

    def reset(self, reset_obs):
        self._reward_rule.reset(reset_obs)

    def __repr__(self):
        return "SPR " + str(self._reward_rule) + "{:.1f}".format(self._scale_k)
