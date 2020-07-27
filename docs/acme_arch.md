# ACME Learning

----
Table of contents
=================

<!--ts-->
   * [Environment](#environment)
      * [TimeStep](#timestep)
      * [Environment Interface](#environment-interface)
      * [Environment Loop](#environmentloop)
      * [ACME Environment](#acme-environment)
   * [Agent](#agent)
      * [Actor](#actor)
      * [Learner](#learner)
   * [Replay Buffer](#replay-buffer)
      * [Adder](#adder)
      * [DataSet](#dataset)
   * [Other Components in ACME](#other-components-in-acme)
      * [Networks](#networks)
      * [Losses](#losses)
<!--te-->

## Environment

ACME adopts `dm_env.Environment` interface. The environment interacts with [*Agent*/*Actor*](#Agent), which takes action from *Agent* and responds with a `TimeStep`.

### TimeStep

A `TimeStep` is a named tuple containing fields

``` none
step_type, observation, discount and reward.
```

An **episode** consists of a series of `TimeStep`s. The step_type of `TimeStep`:
 
* `FIRST`: initial step of an episode. It contains *observation* only and all other fields are `None`.
* `MID`: intermediate step of an episode. Its fields all have values.
* `LAST`: terminal step of an episode. For *finite-horizon* RL settings, it has *discount* as 0.  

### Environment Interface

* `reset()`: This takes no arguments, forces the start of a new episode and returns the first `TimeStep`

* `step(action)`: This takesan `action` parameter and returns a `TimeStep`. If the step is terminal step, then a last `TimeStep` is returned. 

* `action_spec()`, `observation_spec()`, `reward_spec()`, `discount_spec()`: These functions provide environment specifications of the action and the fields of each `TimeStep`. The spec includes `shape`, `dtype` and other attributes, i.e. `max value`, `min value` and etc. These specs are used when constructing the *Agent* in order to make it adpats the environment. `dm_env` provides some concrete spec types, such as `BoundedArray`, `DiscreteArray`.

### EnvironmentLoop

The interaction between *Agent* and *Environment* happens in an environment loop. Here is an example of environment loop:

```python
for _ in range(num_episodes):

    timestep = env.reset()

    while not timestep.last():
        action = agent.select_action(timestep.observation)
        timestep = env.step(action)
```

*Note: the agent simply interacts with env with given policy in the example and hasn't taken learning steps to enhance the policy* 

<img src="diagrams/environment_loop.png" style="max-width:100%;">

### ACME Environment

#### Environment Wrapper

ACME provides a number of `dm_env` environment wrapper, under `acme.wrappers` module, for rendering the environment specs, a usefule one is `SinglePrecisionWrapper` converting any double-precision `float` and `int` components to single-precision.  

#### Environment Loop

ACME implements a RL environment loop class `EnvironmentLoop` which encapsolutes the interaction loop. We can simply construct an `EnvironmentLoop` by passing the `Agent` and `Environment`, then call `run(num_episodes)` to trigger the loop.

----

## Agent

An *Agent* built with Acme are written with two separate modulars `Actor` and `Learner`.

* `Actor` is the component interacting with environment (shown in `Environment` section) and generating experience.

* `Learner` is the component sampling experience and training the relevant action-selection policy (typically neural network models).  

There is an intemate facility `Replay Buffer` (backend `reverb`) interacting with both `Actor` and `Learner`. `Replay Buffer` stores experience data generated from `Actor` and samples to `Learner`. More descriptions in [**Replay Buffer**](#Replay-Buffer) section.

<img src="diagrams/agent_loop.png" style="max-width:100%;">

### Actor

As briefly described in [Environment](#Environment), `Actor` is the component of `Agent` that gains experience from `Environment`. An `Actor` has a *behavior policy* model that makes the decision of action selection. A *behavior policy* is typically based on an *optimized policy* learnt from `Learner` plus some exploration functionality. If an [Adder](#Adder) is provided when constructing an `Actor`, it can also insert the experience to `Replay Buffer` by using `Adder` at each actor step.

#### Actor Interface

* `select_action(observation)`: use *behavior policy* maps the observation to action and return the selected action.

* `observe_first(timestep)`: observe FIRST `TimeStep` from environment and use `adder` to insert the observations to `Replay Buffer`.

* `observe(action, next_timestep)`: observe MID and LAST `TimeStep` from environment and use `adder` to insert action and timestep to `Replay Buffer`.

* `update()`: update behavior policy model parameters from leaner copy. In a single process RL setup, this function does not perform any operations, as the policy model is stored in the memory accessible by both `Actor` and `Leaner`. In distributed actors setup, the learnt policy parameters are stored in a data server remotely, this function will synchronize the policy by sending requests to data server.

`Agent` actually is an sub-class of `Actor`, which inherits the interfaces above and is able to interact with `Environment` directly.

### Learner

`Learner` is the component of `Agent` that trains the policy model from experience. An `Learner` has a *policy model* (typlically neural network) to learn. An [DataSet](#Dataset) connecting to `Replay Buffer` is provided when constructing an `Learner`, where it gets training batchs from `Replay Buffer`.  

### Learner Interface

* `step()`: perform an update step of the learner's parameters. This is the primary implementation of a learner. Take DPG agent's learner as an example, in the `step` function, it fetchs samples from `Replay Buffer` in [DataSet](#DataSet). It builds loss functions of critic network, and actor policy netwrok by dpg method. Then update the network parameters by the gradients.

* `run()`: trigger infinity loop for learner in an isolate process. It is used in distributed actors setup.

----

## Replay Buffer

`Replay Buffer` is the data repository storing experiences. ACME uses reverb.Table as the backend data server for replay buffer.

### Adder

As briefly described in [Agent](#Agent), an adder is responsible for transmitting data from actor to a replay buffer. It also handles data processing before the data transmission. This is a gateway for `Actor` to `Replay Buffer`

#### Adder Interface

Adder interface has almost identical functionalities of parts `Actor`'s methods.    In the respect of data transmission, we can think of `Actor` as an a wrapper to `Adder`. `Adder` can be stripped from `Actor` to perform the data injection. Here is an example:

```python
for _ in range(num_episodes):

    timestep = env.reset()
    adder.add_first(timestep)

    while not timestep.last():
        action = agent.select_action(timestep.observation)
        timestep = env.step(action)
        adder.add(action, timestep)
``` 

* `add_first(timestep)`: called at the beginning of each episode and add initial observation to replay.

* `add(action, next_timestep)`: add action and timestep to replay

* `reset()`: resets the adder's buffer.

#### ACME Adders

ACME has exposed some adders in `acme.adders` module. It has:

* `EpisodeAdder`: adds entire episodes as trajectories
* `SequenceAdder`: adds sequences of fixed length.
* `NStepTransitionAdder`: adds a single N-step transition. I.e., D4PG uses this adder with 5-step transitions.

### DataSet

The `DataSet` is traning sample retrived from replay server. It uses tensorflow Dataset pipeline. ACME implements a function to create the `DataSet` in `acme.datasets` module and encapsolates the communication between `DataSet` and replay data server `reverb`. The `Learner` can simply consume the `Dataset` object directly.

## Other Components in ACME

### Networks

ACME provides `acme.tf.networks` module. This module gives parameterized functions or neural networks for constructing policies, value functions, critic, etc. They are typically based on `Sonnet` neural network library from dm (based on tf).

## Losses

ACME provides some commonly-used loss functions (`acme.tf.losses`) that include:

* a **distributional TD loss** for categorical distributions.
* the Deterministic Policy Gradient **(DPG) loss**.
* the Maximum a posteriori Policy Optimization **(MPO) loss**.
* the **Huber loss** for robust regression.
