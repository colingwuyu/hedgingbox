"""DQN bot implementation."""

import copy

from acme import datasets
from acme import specs
from acme.adders import reverb as adders
from acme.utils import loggers
from acme.agents.tf.dqn import learning
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils

from hb.bots import bot
from hb.bots.dqnbot import actor as dqn_actor
from hb.bots.dqnbot import predictor as dqn_predictor

import tensorflow as tf
import trfl
import reverb
import sonnet as snt
import numpy as np


class DQNBot(bot.Bot):
    """DQN bot.

    """

    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
        network: snt.Module,
        batch_size: int = 256,
        prefetch_size: int = 4,
        target_update_period: int = 100,
        samples_per_insert: float = 32.0,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        importance_sampling_exponent: float = 0.2,
        priority_exponent: float = 0.6,
        n_step: int = 5,
        epsilon: tf.Tensor = None,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        pred_episode: int = 1_000,
        observation_per_pred: int = 10_000,
        pred_only: bool = False,
        logger: loggers.Logger = None,
        pred_logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = '~/acme/',
        checkpoint_per_min: float = 30.
    ):
        # Create a replay server to add data to. This uses no limiter behavior in
        # order to allow the Agent interface to handle it.
        # dqn requires action as scalar
        environment_spec.actions._shape = ()
        environment_spec.actions._dtype = np.int32
        replay_table = reverb.Table(
            name=adders.DEFAULT_PRIORITY_TABLE,
            sampler=reverb.selectors.Prioritized(priority_exponent),
            remover=reverb.selectors.Fifo(),
            max_size=max_replay_size,
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=adders.NStepTransitionAdder.signature(environment_spec))
        self._server = reverb.Server([replay_table], port=None)

        # The adder is used to insert observations into replay.
        address = f'localhost:{self._server.port}'
        adder = adders.NStepTransitionAdder(
            client=reverb.Client(address),
            n_step=n_step,
            discount=discount)

        # The dataset provides an interface to sample from replay.
        replay_client = reverb.TFClient(address)
        dataset = datasets.make_reverb_dataset(
            client=replay_client,
            environment_spec=environment_spec,
            batch_size=batch_size,
            prefetch_size=prefetch_size,
            transition_adder=True)

        # Use constant 0.05 epsilon greedy policy by default.
        if epsilon is None:
            epsilon = tf.Variable(0.05, trainable=False)
        policy_network = snt.Sequential([
            network,
            lambda q: trfl.epsilon_greedy(q, epsilon=epsilon).sample(),
        ])

        # Create a target network.
        target_network = copy.deepcopy(network)

        # Ensure that we create the variables before proceeding (maybe not needed).
        tf2_utils.create_variables(network, [environment_spec.observations])
        tf2_utils.create_variables(
            target_network, [environment_spec.observations])

        # Create the actor which defines how we take actions.
        actor = dqn_actor.DQNActor(
            policy_network, environment_spec.actions, adder)
        # Create the predictor which assess performance
        predictor = dqn_predictor.DQNPredictor(
            network, environment_spec.actions, observation_per_pred, pred_logger
        )

        # The learner updates the parameters (and initializes them).
        learner = learning.DQNLearner(
            network=network,
            target_network=target_network,
            discount=discount,
            importance_sampling_exponent=importance_sampling_exponent,
            learning_rate=learning_rate,
            target_update_period=target_update_period,
            dataset=dataset,
            replay_client=replay_client,
            logger=logger,
            checkpoint=checkpoint)

        if checkpoint:
            self._checkpointer = tf2_savers.Checkpointer(
                directory=checkpoint_subpath,
                objects_to_save=learner.state,
                subdirectory='dqn_learner',
                time_delta_minutes=checkpoint_per_min)
        else:
            self._checkpointer = None

        super().__init__(
            actor=actor,
            learner=learner,
            predictor=predictor,
            min_observations=max(batch_size, min_replay_size),
            observations_per_step=float(batch_size) / samples_per_insert,
            pred_episods=pred_episode,
            observations_per_pred=observation_per_pred,
            pred_only=pred_only)

    def update(self):
        super().update()
        if self._checkpointer is not None:
            self._checkpointer.save()
