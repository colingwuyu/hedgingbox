from .bot import *
import json
from typing import Union
import os

def load_json(json_: Union[dict, str], market, log_dir):
    """
    Create d4pg bot from json 
    example:
        {
            "name": "Greek-Delta",
            "agent_type": "Greek",
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
                        "holding": -0.25000,
                        "instrument": "SPX-OTC"
                    }
                ]
            },
            "model_path": "/content/gdrive/My Drive/Projects/RiskFactorSim/SPX/",
            "hedging_strategies": ["EuroDeltaHedgingStrategy"]
        }
    """
    if isinstance(json_, str):
        dict_json = json.loads(json_)
    else:
        dict_json = json_
    if not os.path.exists(dict_json["model_path"]):
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
    strategies = []
    for strategy_str in dict_json["hedging_strategies"]:
        if strategy_str == "EuroDeltaHedgingStrategy":
            strategies.append(EuroDeltaHedgingStrategy)
        elif strategy_str == "EuroGammaHedgingStrategy":
            strategies.append(EuroGammaHedgingStrategy)
        elif strategy_str == "VarianceSwapReplicatingStrategy":
            strategies.append(VarianceSwapReplicatingStrategy)

    agent = GreekHedgeBot(
        portfolio=portfolio,
        name=dict_json["name"],
        environment_spec=spec,
        pred_dir = log_dir,
        label=dict_json["name"],
        hedging_strategies=strategies
    )
    market.add_agent(agent, agent_name)
    return agent
