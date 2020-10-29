from acme import specs
from acme.utils import loggers
from hb.bots import bot
from hb.bots.nohedgingbot import actor as no_hedge_actor
from hb.bots.nohedgingbot import predictor as no_hedge_predictor
from hb.bots import fake_learner


class NoHedgeBot(bot.Bot):
    """Delta Hedging Bot.


    """

    def __init__(self,
                 environment_spec: specs.EnvironmentSpec,
                 portfolio,
                 pred_dir: str = '~/acme/',
                 pred_episode: int = 1_000 
                 ):
        """Initialize the delta hedging bot

        Args:
            environment_spec (specs.EnvironmentSpec): description of the actions, observations, etc.
        """
        # Create the actor
        actor = no_hedge_actor.NoHedgeActor(environment_spec.actions)
        predictor = no_hedge_predictor.NoHedgePredictor(actor, logger_dir=pred_dir)
        learner = fake_learner.FakeLeaner()

        super().__init__(
            actor=actor,
            learner=learner,
            predictor=predictor,
            min_observations=100,
            observations_per_step=1e9,
            pred_episods=pred_episode,
            observations_per_pred=1,
            portfolio=portfolio,
            pred_only=True)
