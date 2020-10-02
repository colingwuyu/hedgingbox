from hb.market_env.rewardrules.pnl_reward import PnLReward
from hb.market_env.rewardrules.sqrpenalty_reward import SquarePenaltyReward
from hb.market_env.rewardrules.exceed_constraint_penalty_reward import ExceedConstraintPenaltyRewardReward

class RewardRuleFactory():
    @staticmethod
    def create(str_reward_rule: str):
        """Create Reward Rule

        Args:
            str_reward_rule (str): 
                "PnLReward"
                "SPR PnLReward 1.0"

        Returns:
            RewardRule: reward rule
        """
        params = str_reward_rule.split(' ')
        if str_reward_rule == "PnLReward":
            return PnLReward()
        if "SPR" == params[0]:
            return SquarePenaltyReward(RewardRuleFactory.create(params[1]), float(params[2]))
        if str_reward_rule == "ExceedConstraintPenaltyRewardReward":
            return ExceedConstraintPenaltyRewardReward()
