from acme.utils import loggers
from acme import types
import dm_env

from hb import core
from hb.bots.nohedgingbot import actor


class NoHedgePredictor(core.Predictor):
    def __init__(
        self,
        actor: actor.NoHedgeActor,
        logger_dir: str = '~/acme/no_hedge_predictor',
        label: str = 'no_hedge_predictor'
    ):
        super().__init__(actor, 0, logger_dir=logger_dir, label=label)

