from .bot import *
import json
from acme.tf import networks as tf2_networks
from acme import specs
from typing import Mapping, Sequence
from hb.market_env.portfolio import Portfolio
import os


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
    if not os.path.exists(dict_json["model_path"]):
        os.makedirs(dict_json["model_path"])
    model_chkpt = dict_json["model_path"] + "checkpoint"
    if not os.path.exists(model_chkpt):
        os.makedirs(model_chkpt)
    portfolio = Portfolio.load_json(dict_json["portfolio"])
    for position in portfolio.get_portfolio_positions():
        position.set_instrument(
            market.get_instrument(position.get_instrument()))
    portfolio.classify_positions()
    agent_name = dict_json["name"]
    market.add_portfolio(portfolio, dict_json["name"])
    agent_market = market.get_agent_market(agent_name)
    spec = specs.make_environment_spec(agent_market)
    d4pg_networks = make_networks(spec.actions,
                                  dict_json["networks"]["policy_layer_sizes"],
                                  dict_json["networks"]["critic_layer_sizes"],
                                  dict_json["networks"]["vmin"],
                                  dict_json["networks"]["vmax"],
                                  dict_json["networks"]["num_atoms"]
                                  )
    num_prediction_episodes = market.get_validation_episodes()
    observation_per_pred = market.get_train_episodes()
    d4pg_bot = D4PGBot(
        name=agent_name,
        environment_spec=spec,
        policy_network=d4pg_networks['policy'],
        critic_network=d4pg_networks['critic'],
        # policy_optimizer=tf.keras.optimizers.Adam(learning_rate),
        # critic_optimizer=tf.keras.optimizers.Adam(learning_rate),
        risk_obj_func=dict_json["parameters"]["risk_obj_func"],
        risk_obj_c=dict_json["parameters"]["risk_obj_c"],
        exp_mu=dict_json["parameters"]["exp_mu"],
        batch_size=dict_json["parameters"]["batch_size"],
        n_step=dict_json["parameters"]["n_step"],
        sigma=dict_json["parameters"]["sigma"],
        samples_per_insert=dict_json["parameters"]["samples_per_insert"],
        pred_episode=num_prediction_episodes,
        observation_per_pred=observation_per_pred,
        pred_dir=log_dir,
        checkpoint_subpath=dict_json["model_path"],
        checkpoint_per_min=dict_json["parameters"]["checkpoint_per_min"],
        portfolio=portfolio,
        trainable=trainable
    )
    market.add_agent(d4pg_bot, agent_name)
    return d4pg_bot
