from acme.utils import loggers
from acme import types
from acme.agents.tf import actors
import sonnet as snt
import dm_env
from hb import core
from hb.market_env import market_specs


class D4PGPredictor(core.Predictor):
    def __init__(
        self,
        network: snt.Module,
        action_spec: market_specs.DiscretizedBoundedArray,
        num_train_per_pred: int,
        logger_dir: str = '~/acme/d4pg_predictor',
        lable: str = 'd4pg_predictor'
    ):
        pred_actor = actors.FeedForwardActor(network)
        super().__init__(pred_actor, num_train_per_pred, logger_dir, lable)
