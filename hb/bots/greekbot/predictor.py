from acme.utils import loggers
from acme import types
import dm_env

from hb import core
from hb.bots.greekbot import actor


class GreekHedgePredictor(core.Predictor):
    def __init__(
        self,
        actor: actor.GreekHedgeActor,
        logger_dir: str = '~/acme/greek_hedge_predictor',
        label: str = 'greek_hedge_predictor'
    ):
        super().__init__(actor, 0, logger_dir=logger_dir, label=label)

