
import sonnet as snt
import tensorflow as tf
from acme import specs
import tensorflow_probability as tfp

tfd = tfp.distributions


class ClipToSpecGaussian(snt.Module):
  """Sonnet module for adding clipped Gaussian noise to each output."""

  def __init__(self, stddev: float, action_spec: specs.BoundedArray, name: str = 'clipped_gaussian'):
    super().__init__(name=name)
    self._noise = tfd.Normal(loc=0., scale=stddev)
    self._min = action_spec.minimum
    self._max = action_spec.maximum

  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    output = inputs + self._noise.sample(inputs.shape)*tf.math.sqrt(self._max)
    output = tf.clip_by_value(output, self._min, self._max)

    return output
