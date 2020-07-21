"""QTable Q-learning learner implementation."""

from hb.bots.qtablebot import qtable
from hb.bots import DEBUG_PRINT
import time
from typing import Dict, List

import acme
from acme.utils import counting
from acme.utils import loggers
import tensorflow as tf


class QTableLearner(acme.Learner):
    """DQN learner.

    This is the learning component of a Qtable bot. It takes a sampling dataset from replay buffer
    as input and implements update Q-table to learn from this dataset.
    """

    def __init__(
        self,
        qtable: qtable.QTable,
        learning_rate: float,
        target_update_period: int = 1,
        dataset: tf.data.Dataset = None,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
    ):
        """Initializes the learner.

        Args:
          qtable: the policy Q-table (the one being optimized), 0-axis is state space, 1-axis is action space
          discount: discount to use for TD updates.
          learning_rate: learning rate for the Q-table update.
          target_update_period: number of learner steps to perform before updating the target Q-table.
            If it is 1, then target Q-table equals training Q-table.
          dataset: dataset to learn from, whether fixed or from a replay buffer (see
            `acme.datasets.reverb.make_dataset` documentation). If it is not None, then replay buffer is used.
          counter: Counter object for (potentially distributed) counting.
          logger: Logger object for writing logs to.
          checkpoint: boolean indicating whether to checkpoint the learner.
        """

        # Internalise bot components (replay buffer, networks).
        self._iterator = iter(dataset)
        self._qtable = qtable
        if target_update_period > 1:
            self._target_qtable = qtable.copy()
        else:
            self._target_qtable = qtable
        self._learning_rate = learning_rate

        # Internalise the hyperparameters.
        self._target_update_period = target_update_period

        # Learner state.
        self._num_steps = tf.Variable(0, dtype=tf.int32)

        # Internalise logging/counting objects.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.TerminalLogger(
            'learner', time_delta=1.)

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None

    def _step(self) -> Dict[str, tf.Tensor]:
        """Do a step of qtable update with Q-learning TD error."""
        # Pull out the data needed for updates/priorities.
        inputs = next(self._iterator)
        o_tm1, a_tm1, r_t, d_t, o_t = inputs.data
        avg_td_error = 0.
        for i in range(o_tm1.shape[0]):
            o_tm1_i = o_tm1[i]
            a_tm1_i = a_tm1[i]
            r_t_i = r_t[i]
            d_t_i = d_t[i]
            o_t_i = o_t[i]
            cur_q = self._qtable.getQ(o_tm1_i.numpy(), a_tm1_i.numpy())
            target_q = r_t_i + d_t_i * \
                self._target_qtable.select_maxQ(o_t_i.numpy())
            td_error = target_q - cur_q
            avg_td_error += td_error
            inc = self._learning_rate * td_error
            self._qtable.update(o_tm1_i.numpy(), a_tm1_i.numpy(), inc)
            if DEBUG_PRINT:
                print(f"action = {a_tm1_i.numpy()[0]}; observation = {o_tm1_i.numpy()}; cur_Q = {cur_q}; target_Q = {target_q}; inc = {inc}")
        avg_td_error = avg_td_error/o_tm1.shape[0]

        # Periodically update the target network.
        if (self._target_update_period > 1) and (tf.math.mod(self._num_steps, self._target_update_period) == 0):
            self._target_qtable = self._qtable.copy()
        self._num_steps.assign_add(1)

        # Report loss & statistics for logging.
        results = {
            'avg_td_error': avg_td_error,
        }

        return results

    def step(self):
        # Do a batch of SGD.
        result = self._step()

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        result.update(counts)
        self._logger.write(result)

    def get_qtable(self) -> qtable.QTable:
        return self._qtable

    def get_variables(self, names: List[str]) -> List[qtable.QTable]:
        return [self._qtable]

    @ property
    def state(self):
        """Returns the stateful parts of the learner for checkpointing."""
        return {
            'qtable': self._qtable,
            'target_qtable': self._target_qtable,
            'num_steps': self._num_steps
        }
