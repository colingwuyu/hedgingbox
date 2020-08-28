from acme.utils import loggers
from acme import types
import sonnet as snt
import dm_env
from hb import core
from hb.market_env import market_specs
from hb.bots.dqnbot.actor import DQNActor
import trfl


class DQNPredictor(core.Predictor):
    def __init__(
        self,
        network: snt.Module,
        action_spec: market_specs.DiscretizedBoundedArray,
        num_train_per_pred: int,
        logger_dir: str = '~/acme/dqn_predictor',
        label: str = 'dqn_predictor'
    ):
        policy_network = snt.Sequential([
            network,
            lambda q: trfl.epsilon_greedy(q, epsilon=0.).sample(),
        ])
        pred_actor = DQNActor(policy_network=policy_network,
                              action_spec=action_spec,)
        super().__init__(pred_actor, num_train_per_pred, logger_dir=logger_dir, label=label)
