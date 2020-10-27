from acme import specs
from acme.utils import loggers
from hb.bots import bot
from hb.bots.greekbot import actor as greek_hedge_actor
from hb.bots.greekbot import predictor as greek_hedge_predictor
from hb.bots.greekbot.hedging_strategy import *
from hb.bots import fake_learner
from hb.market_env.portfolio import Portfolio


class GreekHedgeBot(bot.Bot):
    """Greek Hedging Bot.


    """

    def __init__(self, portfolio: Portfolio,
                 environment_spec: specs.EnvironmentSpec,
                 hedging_strategies = [EuroDeltaHedgingStrategy, VarianceSwapReplicatingStrategy],
                 pred_dir: str = '~/acme/',
                 label: str = "greek_hedging",
                 pred_episode: int = 1_000 
                 ):
        """Initialize the delta hedging bot

        Args:
            portfolio (Portfolio): portfolio of hedging Stocks and OTC European Options 
            environment_spec (specs.EnvironmentSpec): description of the actions, observations, etc.
        """
        # Create the actor
        actor = greek_hedge_actor.GreekHedgeActor(portfolio, environment_spec.actions, hedging_strategies)
        predictor = greek_hedge_predictor.GreekHedgePredictor(actor, logger_dir=pred_dir, label=label)
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
