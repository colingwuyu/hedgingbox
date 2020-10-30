
from acme import specs
from acme.tf import networks as tf2_networks
from acme.tf import utils as tf2_utils
from acme.tf import savers as tf2_savers
from acme.agents.tf import actors
from acme.utils import counting
from acme.utils import loggers
from acme import types
import numpy as np
import tensorflow as tf
import sonnet as snt
from typing import Mapping, Sequence, Union
from hb.bots import fake_learner
from hb.bots import bot
from hb.bots.d4pgbot import predictor as d4pg_predictor
from hb.market_env.portfolio import Portfolio


def make_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    vmin: float = -150.,
    vmax: float = 150.,
    num_atoms: int = 51,
) -> Mapping[str, types.TensorTransformation]:
    """Creates networks used by the agent."""

    # Get total number of action dimensions from action spec.
    num_dimensions = np.prod(action_spec.shape, dtype=int)

    # Create the shared observation network; here simply a state-less operation.
    observation_network = tf2_utils.batch_concat

    # Create the policy network.
    policy_network = snt.Sequential([
        tf2_networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
        tf2_networks.NearZeroInitializedLinear(num_dimensions),
        tf2_networks.TanhToSpec(action_spec),
    ])

    # Create the critic network.
    critic_network = snt.Sequential([
        # The multiplexer concatenates the observations/actions.
        tf2_networks.CriticMultiplexer(),
        tf2_networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
        tf2_networks.DiscreteValuedHead(vmin, vmax, num_atoms),
    ])

    return {
        'policy': policy_network,
        'critic': critic_network,
        'observation': observation_network,
    }

class D4PGBot(bot.Bot):
    """D4PG Bot.

    """
    def __init__(self,
                name: str,
                portfolio: Portfolio, 
                environment_spec: specs.EnvironmentSpec,
                policy_network: snt.Module,
                critic_network: snt.Module,
                observation_network: types.TensorTransformation = tf.identity,
                discount: float = 1.0,
                pred_episode: int = 1_000,
                observation_per_pred: int = 10_000,
                pred_only: bool = False,
                batch_size: int = 256,
                prefetch_size: int = 4,
                target_update_period: int = 100,
                policy_optimizer: Union[snt.Optimizer,tf.keras.optimizers.Optimizer] = None,
                critic_optimizer: Union[snt.Optimizer,tf.keras.optimizers.Optimizer] = None,
                min_replay_size: int = 1000,
                max_replay_size: int = 1000000,
                samples_per_insert: float = 32.0,
                n_step: int = 5,
                sigma: float = 0.3,
                clipping: bool = True,
                risk_obj_func: bool = True,
                risk_obj_c: np.float32 = 1.5,
                logger: loggers.Logger = None,
                counter: counting.Counter = None,
                pred_dir: str = '~/acme/',
                checkpoint: bool = True,
                checkpoint_subpath: str = '~/acme/',
                checkpoint_per_min: float = 30.,
                replay_table_name: str = adders.DEFAULT_PRIORITY_TABLE):
        # Create a replay server to add data to. This uses no limiter behavior in
        # order to allow the Agent interface to handle it.
        # Make sure observation network is a Sonnet Module.
        observation_network = tf2_utils.to_sonnet_module(observation_network)

        # Create target networks.
        target_policy_network = copy.deepcopy(policy_network)
        target_critic_network = copy.deepcopy(critic_network)
        target_observation_network = copy.deepcopy(observation_network)

        # Get observation and action specs.
        act_spec = environment_spec.actions
        obs_spec = environment_spec.observations
        emb_spec = tf2_utils.create_variables(observation_network, [obs_spec])

        # Create the behavior policy.
        behavior_network = snt.Sequential([
            observation_network,
            policy_network,
            tf2_networks.ClippedGaussian(sigma),
            tf2_networks.ClipToSpec(act_spec),
        ])

        # Create variables.
        tf2_utils.create_variables(policy_network, [emb_spec])  # pytype: disable=wrong-arg-types
        tf2_utils.create_variables(critic_network, [emb_spec, act_spec])
        tf2_utils.create_variables(target_policy_network, [emb_spec])  # pytype: disable=wrong-arg-types
        tf2_utils.create_variables(target_critic_network, [emb_spec, act_spec])
        tf2_utils.create_variables(target_observation_network, [obs_spec])

        # Create the actor which defines how we take actions.
        actor = actors.FeedForwardActor(behavior_network, adder=adder)

        # Create the predictor 
        pred_behavior_network = snt.Sequential([
            observation_network,
            policy_network,
            tf2_networks.ClipToSpec(act_spec),
        ])
        predictor = d4pg_predictor.D4PGPredictor(network=pred_behavior_network,
                                                action_spec=act_spec, 
                                                num_train_per_pred=observation_per_pred, 
                                                logger_dir=pred_dir,
                                                risk_obj=risk_obj_func,
                                                risk_obj_c=risk_obj_c)
        learner = fake_learner.FakeLeaner()
        # Store online and target networks.
        self._policy_network = policy_network
        self._critic_network = critic_network
        self._target_policy_network = target_policy_network
        self._target_critic_network = target_critic_network

        # Make sure observation networks are snt.Module's so they have variables.
        self._observation_network = tf2_utils.to_sonnet_module(observation_network)
        self._target_observation_network = tf2_utils.to_sonnet_module(
            target_observation_network)

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        

        if checkpoint:
            self._checkpointer = tf2_savers.Checkpointer(
                directory=checkpoint_subpath,
                objects_to_save=self.state,
                subdirectory='d4pg_learner',
                time_delta_minutes=checkpoint_per_min,
                add_uid=False)
        else:
            self._checkpointer = None
            
        super().__init__(
            name=name,
            actor=actor,
            learner=learner,
            predictor=predictor,
            min_observations=max(batch_size, min_replay_size),
            observations_per_step=float(batch_size) / samples_per_insert,
            pred_episods=pred_episode,
            observations_per_pred=observation_per_pred,
            portfolio=portfolio,
            pred_only=pred_only)

    def update(self):
        super().update()
        if (self._checkpointer is not None) and self._predictor.is_best_perf():
            self._checkpointer.save(force=True)

    @property
    def state(self):
        return {
                'policy': self._policy_network,
                'critic': self._critic_network,
                'observation': self._observation_network,
                'target_policy': self._target_policy_network,
                'target_critic': self._target_critic_network,
                'target_observation': self._target_observation_network
            }
