"""QTable bot implementation."""

from acme import datasets
from acme import specs
from acme.adders import reverb as adders
from acme.agents import agent
from acme.utils import loggers

from hb.bots import bot
from hb.bots.qtablebot import actor as qtable_actor
from hb.bots.qtablebot import predictor as qtable_predictor
from hb.bots.qtablebot import learning
from hb.bots.qtablebot import qtable

import reverb
import pickle


class QTableBot(bot.Bot):
    """QTable bot.

    This implements a single-process QTable bot. This is a simple Q-learning
    algorithm that inserts N-step transitions into a replay buffer, and
    periodically updates its Q-Table policy by sampling these transitions.
    """

    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
        batch_size: int = 1,
        prefetch_size: int = None,
        target_update_period: int = 1,
        samples_per_insert: float = 1.,
        min_replay_size: int = 1,
        max_replay_size: int = 1,
        n_step: int = 1,
        epsilon: float = 0.1,
        learning_rate: float = 1e-3,
        discount: float = 1.0,
        pred_episode: int = 1_000,
        observation_per_pred: int = 10_000,
        pred_only: bool = False,
        logger: loggers.Logger = None,
        pred_dir: str = '~/acme/',
        checkpoint: bool = True,
        checkpoint_subpath: str = '~/acme/',
    ):
        """Initialize the agent.

        Args:
          environment_spec: description of the actions, observations, etc.
          batch_size: batch size for updates.
          prefetch_size: size to prefetch from replay.
          target_update_period: number of learner steps to perform before updating
            the target networks.
          samples_per_insert: number of samples to take from replay for every insert
            that is made.
          min_replay_size: minimum replay size before updating. This and all
            following arguments are related to dataset construction and will be
            ignored if a dataset argument is passed.
          max_replay_size: maximum replay size.
          n_step: number of steps to squash into a single transition.
          epsilon: probability of taking a random action; ignored if a policy
            network is given.
          learning_rate: learning rate for the q-network update.
          discount: discount to use for TD updates.
          logger: logger object to be used by learner.
          checkpoint: boolean indicating whether to checkpoint the learner.
          checkpoint_subpath: directory for the checkpoint.
        """
        # Create a replay server to add data to. This uses no limiter behavior in
        # order to allow the Agent interface to handle it.
        replay_table_name = 'replay_table'
        replay_table = reverb.Table(
            name=replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=max_replay_size,
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=adders.NStepTransitionAdder.signature(environment_spec))
        self._server = reverb.Server([replay_table], port=None)

        # The adder is used to insert observations into replay.
        address = f'localhost:{self._server.port}'
        adder = adders.NStepTransitionAdder(
            priority_fns={replay_table_name: lambda x: 1.},
            client=reverb.Client(address),
            n_step=n_step,
            discount=discount)

        # The dataset provides an interface to sample from replay.
        dataset = datasets.make_reverb_dataset(
            table=replay_table_name,
            client=reverb.TFClient(address),
            batch_size=batch_size,
            prefetch_size=prefetch_size,
            environment_spec=environment_spec,
            transition_adder=True)

        q_table = qtable.QTable(environment_spec.observations,
                                environment_spec.actions)

        # Create the actor which defines how we take actions.
        actor = qtable_actor.QTableActor(
            qtable=q_table,
            epsilon=epsilon,
            adder=adder
        )
        # Create the predictor which assess performance
        predictor = qtable_predictor.QTablePredictor(
            qtable=q_table,
            num_train_per_pred=observation_per_pred,
            logger_dir=pred_dir
        )

        # The learner updates the parameters (and initializes them).
        learner = learning.QTableLearner(
            qtable=q_table,
            learning_rate=learning_rate,
            target_update_period=target_update_period,
            dataset=dataset,
            logger=logger,
        )

        # if checkpoint:
        #     self._checkpointer = tf2_savers.Checkpointer(
        #         directory=checkpoint_subpath,
        #         objects_to_save=learner.state,
        #         subdirectory='qtable_bot',
        #         time_delta_minutes=60.)
        # else:
        #     self._checkpointer = None

        super().__init__(
            actor=actor,
            learner=learner,
            predictor=predictor,
            min_observations=max(batch_size, min_replay_size),
            observations_per_step=float(batch_size) / samples_per_insert,
            pred_episods=pred_episode,
            observations_per_pred=observation_per_pred,
            pred_only=pred_only)

    def get_qtable(self):
        return self._learner.get_qtable()

    def save(self, location='../ACME Models/QTable.pickle'):
        with open(location, 'wb') as pf:
            pickle.dump(self.get_qtable(), pf)

    def restore(self, location='../ACME Models/QTable.pickle'):
        with open(location, 'rb') as pf:
            qtable = pickle.load(pf)
        self._learner._qtable = qtable
        self._actor._qtable = qtable
        self._predictor._actor._qtable = qtable
        if self._learner._target_update_period > 1:
            self._learner._target_qtable = qtable.copy()
        else:
            self._learner._target_qtable = qtable

    def update(self):
        super().update()
        # if self._checkpointer is not None:
        #     self._checkpointer.save()
