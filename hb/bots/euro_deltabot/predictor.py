from acme.utils import loggers
from acme import types
import dm_env

from hb import core
from hb.bots.euro_deltabot import actor


class DeltaHedgePredictor(core.Predictor):
    def __init__(
        self,
        actor: actor.DeltaHedgeActor,
        logger_dir: str = '~/acme/delta_hedge_predictor',
        label: str = 'delta_hedge_predictor'
    ):
        super().__init__(actor, 0, logger_dir=logger_dir, label=label)

