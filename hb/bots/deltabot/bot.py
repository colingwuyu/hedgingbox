from acme import specs
from hb.bots import bot
from hb.bots.deltabot import actor as delta_hedge_actor
from hb.bots.deltabot import predictor as delta_hedge_predictor
from hb.bots import fake_learner


class DeltaHedgeBot(bot.Bot):
    """Delta Hedging Bot.


    """

    def __init__(self,
                 environment_spec: specs.EnvironmentSpec,
                 pred_episode: int = 1_000 
                 ):
        """Initialize the delta hedging bot

        Args:
            environment_spec (specs.EnvironmentSpec): description of the actions, observations, etc.
        """
        # Create the actor
        actor = delta_hedge_actor.DeltaHedgeActor(environment_spec.actions)
        predictor = delta_hedge_predictor.DeltaHedgePredictor(actor)
        learner = fake_learner.FakeLeaner()

        super().__init__(
            actor=actor,
            learner=learner,
            predictor=predictor,
            min_observations=100,
            observations_per_step=1e9,
            pred_episods=pred_episode,
            observations_per_pred=1,
            pred_only=True)
