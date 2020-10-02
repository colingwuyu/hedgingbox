from hb.market_env.rewardrules import reward_rule
from hb.instrument.cash_account import CashAccount
from hb.market_env.portfolio import Portfolio
from hb.utils.date import get_cur_days
import dm_env
from acme import types
from typing import Dict
import numpy as np


class ExceedConstraintPenaltyRewardReward(reward_rule.RewardRule):
    def __init__(self):
        self._this_step_obs = None
        self._first_reward = True
        self._exceed_penalty = 1e9

    def step_reward(self, step_type: dm_env.StepType,
                    step_pnl: float, 
                    action: types.NestedArray,
                    extra: dict = dict()
                    ) -> types.NestedArray:
        # interest from cash account
        if extra["exceed_constraint"]:
            return step_pnl - self._exceed_penalty
        else:
            return step_pnl

    def reset(self, reset_observation):
        pass

    def __repr__(self):
        return "ExceedConstraintPenaltyRewardReward"
