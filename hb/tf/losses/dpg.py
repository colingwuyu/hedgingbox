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
  """Deterministic policy gradient loss by maxmizing the risk oriented objective function
  F(S_t, A_t) = E[Z(S_t, A_t)] - c*sqrt(Variance(Z(S_t, A_t))) 
  d(loss)/dw = -dF/da*da/dw

  """

  # Calculate the gradient dq/da.
  c = tf.cast(c, q_max.dtype)
  f = q_max - c * tf.sqrt(q_var_max)
  dfda = tape.gradient([f], [a_max])[0]

  if dfda is None:
    raise ValueError('risk obj function needs to be a function of a_max.')
  
  # Clipping the gradient dq/da.
  if dqda_clipping is not None:
    if dqda_clipping <= 0:
      raise ValueError('dqda_clipping should be bigger than 0, {} found'.format(
          dqda_clipping))
    if clip_norm:
      dfda = tf.clip_by_norm(dfda, dqda_clipping, axes=-1)
    else:
      dfda = tf.clip_by_value(dfda, -1. * dqda_clipping, dqda_clipping)
  # Target_a ensures correct gradient calculated during backprop.
  target_a = dfda + a_max
  # Stop the gradient going through Q network when backprop.
  target_a = tf.stop_gradient(target_a)
  # Gradient only go through actor network.
  loss = 0.5 * tf.reduce_sum(tf.square(target_a - a_max), axis=-1)
  # This recovers the DPG because (letting w be the actor network weights):
  # d(loss)/dw = 0.5 * (2 * (target_a - a_max) * d(target_a - a_max)/dw)
  #            = (target_a - a_max) * [d(target_a)/dw  - d(a_max)/dw]
  #            = df/da * [d(target_a)/dw  - d(a_max)/dw]  # by defn of target_a
  #            = df/da * [0 - d(a_max)/dw]                # by stop_gradient
  #            = - df/da * da/dw

  return loss
