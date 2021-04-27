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
        risk_obj: bool = False,
        risk_obj_c: float = 1.5,
        mu_lambda: float = 1.0,
        logger_dir: str = '~/acme/d4pg_predictor',
        label: str = 'd4pg_predictor'
    ):
        pred_actor = actors.FeedForwardActor(network)
        super().__init__(pred_actor, num_train_per_pred, risk_obj, risk_obj_c, mu_lambda, logger_dir, label)
