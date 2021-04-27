from typing import Sequence

from acme import types
from acme.tf import utils as tf2_utils
from acme.tf.networks import base
import sonnet as snt
import tensorflow as tf

class LayerConstraint(snt.Module):
  """Simple feedforward MLP torso with initial layer-norm.

  This module is an MLP which uses LayerNorm (with a tanh normalizer) on the
  first layer and non-linearities (elu) on all but the last remaining layers.
  """

  def __init__(self, layer_sizes: Sequence[int], activate_final: bool = False):
    """Construct the MLP.

    Args:
      layer_sizes: a sequence of ints specifying the size of each layer.
      activate_final: whether or not to use the activation function on the final
        layer of the neural network.
    """
    super().__init__(name='feedforward_mlp_torso')

    self._network = snt.Sequential([
        snt.Linear(layer_sizes[0], w_init=uniform_initializer),
        snt.LayerNorm(
            axis=slice(1, None), create_scale=True, create_offset=True),
        tf.nn.tanh,
        snt.nets.MLP(
            layer_sizes[1:],
            w_init=uniform_initializer,
            activation=tf.nn.elu,
            activate_final=activate_final),
    ])

  def __call__(self, observations: types.Nest) -> tf.Tensor:
    """Forwards the policy network."""
    return self._network(tf2_utils.batch_concat(observations))
