from .bot import *
import json
from acme.tf import networks as tf2_networks
import tensorflow as tf
from acme import specs
from typing import Mapping, Sequence
from hb.market_env.portfolio import Portfolio
from hb.utils.container_util import get_value
from hb.tf.savers import load_bot
import os

DEFAULT_JSON = {
    "networks": {
        "policy_layer_sizes": [256, 256, 256],
        "critic_layer_sizes": [512, 512, 256],
        "vmin": -150,
        "vmax": 150,
        "num_atoms": 51
    },
    "parameters": {
        "risk_obj_func": True,
        "risk_obj_c": 1.0,
        "mu_lambda": 0,
        "batch_size": 256,
        "n_step": 5,
        "sigma": 0.08,
        "samples_per_insert": 32,
        "checkpoint_per_min": 30
    }
}


def make_networks(
    bot_name: str,
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

    with tf.name_scope(bot_name):
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


def load_json(json_: Union[dict, str], market, log_dir, trainable=True):
    """Create d4pg bot from json
    example:
    {
        "name": "D4PG-1",
        "agent_type": "D4PG",
        "portfolio": {
            "risk_limits": {
                "delta": [-3, 3]
            },
            "positions": [
                {
                    "holding": -1.30403,
                    "trading_limit": [-5,5],
                    "instrument": "SPX"
                },
                {
                    "holding": 0,
                    "trading_limit": [-10,10],
                    "instrument": "SPX-Listed"
                },
                {
                    "holding": -0.25000,
                    "instrument": "SPX-OTC"
                }
            ]
        },
        "model_path": "/content/gdrive/My Drive/Projects/RiskFactorSim/SPX/",
        "networks": {
            "policy_layer_sizes": [256, 256, 256],
            "critic_layer_sizes": [512, 512, 256],
            "vmin": -150,
            "vmax": 150,
            "num_atoms": 51
        },
        "parameters": {
            "risk_obj_func": true,
            "risk_obj_c": 1.5,
            "batch_size": 256,
            "n_step": 5,
            "samples_per_insert": 32,
            "observation_per_pred": 1000,
            "checkpoint_per_min": 30
        }
    }
    Args:
        json_ (Union[dict, str]): [description]
    """
    if isinstance(json_, str):
        dict_json = json.loads(json_)
    else:
        dict_json = json_

    trainable_variables_snapshot = {}

    if '.bot' in dict_json["model_path"]:
        model_path = dict_json["model_path"]
        nets, dict_json = load_bot(dict_json["model_path"])
        dict_json["model_path"] = model_path
        policy_network_snapshot = nets['policy']
        critic_network_snapshot = nets['critic']
        trainable_variables_snapshot['policy'] = {}
        for var in policy_network_snapshot.trainable_variables:
            trainable_variables_snapshot['policy']['/'.join(
                var.name.split('/')[1:])] = var.numpy()
        trainable_variables_snapshot['critic'] = {}
        for var in critic_network_snapshot.trainable_variables:
            trainable_variables_snapshot['critic']['/'.join(
                var.name.split('/')[1:])] = var.numpy()

    if 'pretrained_model' in dict_json:
        nets, pretrained_dict_json = load_bot(dict_json["pretrained_model"])
        for k in DEFAULT_JSON['networks'].keys():
            assert get_value(dict_json, DEFAULT_JSON, "networks", k) == get_value(
                pretrained_dict_json, DEFAULT_JSON, "networks", k), "Pretrained Model's  architecture is not the same."
        policy_network_snapshot = nets['policy']
        critic_network_snapshot = nets['critic']
        trainable_variables_snapshot['policy'] = {}
        for var in policy_network_snapshot.trainable_variables:
            trainable_variables_snapshot['policy']['/'.join(
                var.name.split('/')[1:])] = var.numpy()
        trainable_variables_snapshot['critic'] = {}
        for var in critic_network_snapshot.trainable_variables:
            trainable_variables_snapshot['critic']['/'.join(
                var.name.split('/')[1:])] = var.numpy()

    if ('.bot' not in dict_json["model_path"]) \
            and (not os.path.exists(dict_json["model_path"])):
        os.makedirs(dict_json["model_path"])

    portfolio = Portfolio.load_json(dict_json["portfolio"])
    for position in portfolio.get_portfolio_positions():
        position.set_instrument(
            market.get_instrument(position.get_instrument()))
    portfolio.classify_positions()
    agent_name = dict_json["name"]
    market.add_portfolio(portfolio, dict_json["name"])
    agent_market = market.get_agent_market(agent_name)
    spec = specs.make_environment_spec(agent_market)

    d4pg_networks = make_networks(agent_name,
                                  spec.actions,
                                  get_value(dict_json, DEFAULT_JSON, "networks",
                                            "policy_layer_sizes"),
                                  get_value(dict_json, DEFAULT_JSON, "networks",
                                            "critic_layer_sizes"),
                                  get_value(dict_json, DEFAULT_JSON, "networks",
                                            "vmin"),
                                  get_value(dict_json, DEFAULT_JSON, "networks",
                                            "vmax"),
                                  get_value(dict_json, DEFAULT_JSON, "networks",
                                            "num_atoms")
                                  )

    num_validation_episodes = market.get_validation_episodes()
    observation_per_pred = market.get_train_episodes()
    d4pg_bot = D4PGBot(
        name=agent_name,
        environment_spec=spec,
        bot_json=dict_json,
        snapshot_variables=trainable_variables_snapshot,
        policy_network=d4pg_networks['policy'],
        critic_network=d4pg_networks['critic'],
        # policy_optimizer=tf.keras.optimizers.Adam(learning_rate),
        # critic_optimizer=tf.keras.optimizers.Adam(learning_rate),
        risk_obj_func=get_value(dict_json, DEFAULT_JSON,
                                "parameters", "risk_obj_func"),
        mu_lambda=get_value(dict_json, DEFAULT_JSON,
                            "parameters", "mu_lambda"),
        batch_size=get_value(dict_json, DEFAULT_JSON,
                             "parameters", "batch_size"),
        n_step=get_value(dict_json, DEFAULT_JSON, "parameters", "n_step"),
        sigma=get_value(dict_json, DEFAULT_JSON, "parameters", "sigma"),
        samples_per_insert=get_value(
            dict_json, DEFAULT_JSON, "parameters", "samples_per_insert"),
        pred_episode=0,
        validation_episodes=num_validation_episodes,
        observation_per_pred=observation_per_pred,
        pred_dir=log_dir,
        checkpoint_subpath=dict_json["model_path"],
        checkpoint_per_min=get_value(
            dict_json, DEFAULT_JSON, "parameters", "checkpoint_per_min"),
        portfolio=portfolio,
        trainable=trainable
    )
    market.add_agent(d4pg_bot, agent_name)
    return d4pg_bot
