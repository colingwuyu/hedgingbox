from acme.utils import loggers
from acme import types
import sonnet as snt
import dm_env
from hb import core
from hb.market_env import market_specs
from hb.bots.dqnbot.actor import DQNActor


class DQNPredictor(core.Predictor):
    def __init__(
        self,
        network: snt.Module,
        action_spec: market_specs.DiscretizedBoundedArray,
        num_train_per_pred: int,
        logger: loggers.Logger = None,
        lable: str = 'dqn_predictor'
    ):
        pred_actor = DQNActor(policy_network=network,
                              action_spec=action_spec,)
        super().__init__(pred_actor, num_train_per_pred, logger, lable)
