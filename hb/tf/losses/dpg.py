import tensorflow as tf


def risk_dpg(
    q_max: tf.Tensor, # q mean
    q_var_max: tf.Tensor, # q variance
    c: tf.Tensor, # coefficient of standard deviation
    a_max: tf.Tensor,
    tape: tf.GradientTape,
    dqda_clipping: float = None,
    clip_norm: bool = False,
) -> tf.Tensor:
  """Deterministic policy gradient loss by minimizing the risk oriented objective function
  F(S_t, A_t) = -E[Z(S_t, A_t)] + c*sqrt(Variance(Z(S_t, A_t))) 
  d(loss)/dw = -dF/da*da/dw

  """

  # Calculate the gradient dq/da.
  dqda = tape.gradient([q_max], [a_max])[0]

  # Calculate the gradient dq_var/da
  dqvarda = tape.gradient([q_var_max], [a_max])[0]

  if dqda is None:
    raise ValueError('q_max needs to be a function of a_max.')
  if dqvarda is None:
    raise ValueError('q_var_max needs to be a function of a_max')

  # Clipping the gradient dq/da.
  if dqda_clipping is not None:
    if dqda_clipping <= 0:
      raise ValueError('dqda_clipping should be bigger than 0, {} found'.format(
          dqda_clipping))
    if clip_norm:
      dqda = tf.clip_by_norm(dqda, dqda_clipping, axes=-1)
      dqvarda = tf.clip_by_norm(dqvarda, dqda_clipping, axes=-1)
    else:
      dqda = tf.clip_by_value(dqda, -1. * dqda_clipping, dqda_clipping)
  c = tf.cast(c, dqda.dtype)
  dfda = -dqda + 0.5*c*tf.pow(q_var_max, -0.5)*dqvarda
  # Target_a ensures correct gradient calculated during backprop.
  target_a = dfda + a_max
  # Stop the gradient going through Q network when backprop.
  target_a = tf.stop_gradient(target_a)
  # Gradient only go through actor network.
  loss = - 0.5 * tf.reduce_sum(tf.square(target_a - a_max), axis=-1)

  return loss
