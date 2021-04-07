from acme.agents import agent
from hb.instrument.instrument_factory import InstrumentFactory
from hb.instrument.stock import Stock
from hb.instrument.european_option import EuropeanOption
from hb.instrument.variance_swap import VarianceSwap
from hb.instrument.cash_account import CashAccount
from hb.market_env.rewardrules.pnl_reward import PnLReward
from hb.market_env import market_specs
from hb.market_env.rewardrules import reward_rule
from hb.market_env.rewardrules.reward_rule_factory import RewardRuleFactory
from hb.utils.date import *
from hb.utils.process import *
from hb.utils.heston_calibration import *
from hb.utils.consts import *
from hb.market_env.portfolio import Portfolio
from hb.riskfactorsimulator.simulator import Simulator
from hb.utils.handler import Handler

from acme import wrappers
import math
from typing import List, Union
import numpy as np
import dm_env
from dm_env import specs
from os.path import join as pjoin
import json


class EpisodeCounter:
    def __init__(self, total_paths=None, total_steps=None):
        """episode path and step counter

        Args:
            total_paths (float, optional): total paths. Defaults to None.
            total_steps (float, optional): total steps including time 0 in an episode. Defaults to None.
        """
        self._path_counter = -1
        self._step_counter = -1
        self._total_paths = total_paths
        self._total_steps = total_steps
        
    def set_total_paths(self, total_paths):
        self._total_paths = total_paths
    
    def get_total_paths(self):
        return self._total_paths
    
    def total_paths(self, total_paths):
        self._total_paths = total_paths
        return self

    def set_total_steps(self, total_steps):
        self._total_steps = total_steps

    def get_total_steps(self):
        return self._total_steps
    
    def total_steps(self, total_steps):
        self._total_steps = total_steps
        return self

    def inc_path_counter(self):
        self._path_counter += 1
        
    def inc_step_counter(self):
        self._step_counter += 1
        
    def get_path_step(self):
        return self._path_counter % self._total_paths, self._step_counter % (self._total_steps+1)
    
    def reset(self):
        self._path_counter = -1
        self._step_counter = -1

class SingleAgentMarket(dm_env.Environment):
    __slots__ = ["num_steps", 
                 "cash_account", "reward_rule", "hedging_step_in_days"
                 "portfolio", "event_trans_cost"]

    def __init__(self, ir, portfolio: Portfolio, reward_rule, hedging_step_in_days):
        self.cash_account = CashAccount(interest_rates=ir)
        self.hedging_step_in_days = hedging_step_in_days
        self.set_portfolio(portfolio)
        self.reward_rule = reward_rule
        self.event_trans_cost = 0.

    def set_portfolio(self, portfolio: Portfolio):
        """set the trading portfolio into market

        Args:
            portfolio (Portfolio): A portfolio that contains a list of exotic instruments (liabilities) 
                                   and a list of eligible hedging instruments
        """
        num_steps = 0
        for derivative in portfolio.get_liability_portfolio():
            num_steps_to_maturity = int(days_from_time(derivative.get_instrument().get_maturity_time()) / self.hedging_step_in_days)
            num_steps = max(num_steps, num_steps_to_maturity)
        self.portfolio = portfolio 
        self.num_steps = num_steps
    
    def reset(self):
        """dm_env interface
           Reset episode
        
        Returns:
            [dm_env.TimeStep]: FIRST TimeStep
        """
        self.portfolio.reset()
        self.cash_account.reset()
        initial_cashflow = self.portfolio.get_nav()
        # save initial cashflow into cash account
        self.cash_account.add(-initial_cashflow)
        self.reward_rule.reset(self.portfolio)
        # with open('logger.csv', 'a') as logger:
        #     logger.write(','.join([str(k) for k in 
        #         [get_cur_days(), 0., 100., 5, 
        #         self._portfolio.get_hedging_portfolio()[0].get_instrument().get_price()[0], 
        #         self._portfolio.get_hedging_portfolio()[0].get_holding(),
        #         self._portfolio.get_liability_portfolio()[0].get_instrument().get_price(), 
        #         self._portfolio.get_liability_portfolio()[0].get_holding(), 
        #         self._cash_account.get_balance(), 
        #         0., '', 0., '']])+'\n'
        #     )
        return dm_env.restart(np.append(self._observation(), 0.))

    def step(self, action):
        pass

    def _observation(self):
        """Construct state observation
           Now the observation includes
                - all hedging positions' price, holding in the portfolio
                - all derivative positions' price, remaining times in the portfolio
                ## - all hedging positions' holding constraint breaching indicator in the portfolio

        Returns:
            market_observations [np.darray]: a list of state observation
        """
        market_observations = np.array([], dtype=np.float)
        for position in self.portfolio.get_hedging_portfolio():
            price = position.get_instrument().get_price()
            # add position's price and holding
            market_observations = np.append(market_observations, [price, position.get_holding()])
        for position in self.portfolio.get_liability_portfolio():
            market_observations = np.append(market_observations, [position.get_instrument().get_price(), 
                                                                  position.get_instrument().get_remaining_time()])
        # for position in self._portfolio.get_hedging_portfolio():
        #     breach_constraint = position.get_breach_holding_constraint()
        #     market_observations = np.append(market_observations, [1.0 if breach_constraint else 0.0])
        return market_observations
    
    def observation_spec(self):
        """dm_env interface
           observation specification returns the observation shapes
           Now the observation includes
                - all hedging positions' price, holding in the portfolio
                - all derivative positions' price, remaining times in the portfolio
                ## - all hedging positions' holding constraint breaching indicator in the portfolio

        Returns:
            [dm_env.specs.Array]: observation specification
        """
        # positions' price, holding; current time; derivatives' remaining times; cash account balance
        num_obs = 2*len(self.portfolio.get_portfolio_positions())
        obs_shape = (num_obs, )
        return specs.Array(
            shape=obs_shape, dtype=float, name="market_observations"
        )

    def action_spec(self):
        """Returns the action spec.
        """
        maximum = np.ones(len(self.portfolio.get_hedging_portfolio()))
        minimum = -1*maximum
        return specs.BoundedArray(
            shape=(len(maximum),), dtype=float,
            minimum=minimum, maximum=maximum, name="hedging_actions"
        )

class Market(dm_env.Environment):
    __slots__ = ["_name", "_valuation_date",
                 "_training_simulator", "_validation_simulator", "_scenario_simulator", "_current_simulator_handler",
                 "_training_counter", "_validation_counter", "_scenario_counter", "_current_counter_handler",
                 "_reward_rule", "_hedging_step_in_days", "_instruments", "_mode",
                 "_agent_markets", "_agents", "_agent_actions", "_trainable_agent"]

    """Market Environment
    """

    @classmethod
    def load_json(cls, json_: Union[dict, str]):
        """Constructor of market environment
        example:
        {
            "name": "BSM_AMZN_SPX",
            "valuation_date": "2020-02-03" (yyyy-mm-dd),
            "reward_rule": "PnLReward",
            "hedging_step_in_days": 1,
            "validation_rng_seed": 1234,
            "training_episodes": 10000,
            "validation_episodes": 1000,
            "riskfactorsimulator": {
                "ir": 0.015,
                "equity": [
                    {
                        "name": "AMZN",
                        "riskfactors": ["Spot", 
                                        "Vol 3Mx100",
                                        "Vol 2Mx100", 
                                        "Vol 4Wx100"],
                        "process_param": {
                            "process_type": "Heston",
                            "param": {
                                "spot": 100,
                                "spot_var": 0.096024,
                                "drift": 0.25,
                                "dividend": 0.0,
                                "kappa": 6.288453,
                                "theta": 0.397888,
                                "epsilon": 0.753137,
                                "rho": -0.696611
                            } 
                        }
                    },
                    {
                        "name": "SPX",
                        "riskfactors": ["Spot", "Vol 3Mx100"],
                        "process_param": {
                            "process_type": "GBM",
                            "param": {
                                "spot": 100,
                                "drift": 0.10,
                                "dividend": 0.01933,
                                "vol": 0.25
                            } 
                        }
                    }
                ],
                "correlation": [
                    {
                        "equity1": "AMZN",
                        "equity2": "SPX",
                        "corr": 0.8
                    }
                ]
            },
            "instruments" : [
                "Stock SPX 0 0 0.5",
                "EuroOpt SPX Listed 2021-01-15 Put 30 3 (SPX-Listed)",
                "EuroOpt SPX OTC 2021-01-15 Call 45 3 (SPX-OTC)",
                "VarSwap SPX 3M 10.65 0.1 (SPX_3M_VAR_SWAP)",
                "EuroOpt SPX Listed 3M Call 120 0.5 2.5 (SPX_LISTED_CALL1)",
                "EuroOpt SPX Listed 3M Call 100 0.5 2.5 (SPX_LISTED_CALL2)",
                "EuroOpt SPX Listed 3M Put 100 0.5 2.5 (SPX_LISTED_PUT1)",
                "EuroOpt SPX Listed 3M Put 80 0.5 2.5 (SPX_LISTED_PUT2)"
            ],
            "extra": [
                {
                    "instrument": "SPX_3M_VAR_SWAP",
                    "replicating_opts": ["SPX_LISTED_CALL1", "SPX_LISTED_CALL2", "SPX_LISTED_PUT1", "SPX_LISTED_PUT2"],
                    "excl_realized_var": true
                }
            ]
        }
        Args:
            json_ (Union[dict, str]): market environment in json
        """
        if isinstance(json_, str):
            dict_json = json.loads(json_)
        else:
            dict_json = json_
        market = cls()
        market._name = dict_json["name"]
        market._valuation_date = dict_json["valuation_date"]
        set_valuation_date(date_from_str(dict_json["valuation_date"]))
        market._training_simulator = Simulator.load_json(dict_json["riskfactorsimulator"])
        market._training_counter = EpisodeCounter(dict_json["training_episodes"], dict_json["num_steps_per_episode"]+1)
        market._validation_simulator = Simulator.load_json(dict_json["riskfactorsimulator"]).rng_seed(dict_json["validation_rng_seed"])
        market._validation_counter = EpisodeCounter(dict_json["validation_episodes"], dict_json["num_steps_per_episode"]+1)
        market._scenario_simulator = Simulator.load_json(dict_json["riskfactorsimulator"])
        market._scenario_counter = EpisodeCounter()
        market._current_simulator_handler = Handler(market._training_simulator)
        market._current_counter_handler = Handler(market._training_counter)
        market._cash_account = CashAccount(interest_rates=market._training_simulator.get_ir())
        market._reward_rule = RewardRuleFactory.create(dict_json["reward_rule"])
        market._hedging_step_in_days = dict_json["hedging_step_in_days"]
        market._training_simulator = market._training_simulator.time_step(market._hedging_step_in_days/DAYS_PER_YEAR)\
                                                               .num_steps(dict_json["num_steps_per_episode"]+1)
        market._validation_simulator = market._validation_simulator.time_step(market._hedging_step_in_days/DAYS_PER_YEAR)\
                                                                   .num_steps(dict_json["num_steps_per_episode"]+1)
        ###################################################################
        # Instruments
        ###################################################################
        market._instruments = {}
        # create instrument
        for inst in dict_json["instruments"]:
            inst_obj = InstrumentFactory.create(inst) 
            market._instruments[inst_obj.get_name()] = inst_obj
        # set underlying to derivatives
        for inst_obj in market._instruments.values():
            if not isinstance(inst_obj, Stock):
                inst_obj.set_underlying(
                    market.get_instrument(inst_obj.get_underlying_name())
                )
        # instrument extra arguments
        if "extra" in dict_json:
            extras = dict_json["extra"]
            for extra in extras:
                instrument = market.get_instrument(extra["instrument"])
                if isinstance(instrument, VarianceSwap):
                    repliacting_portfolio = [market.get_instrument(opts) for opts in extra["replicating_opts"]]
                    instrument.replicating_opts(repliacting_portfolio)
                    instrument.set_excl_realized_var(extra["excl_realized_var"])
        # market.set_portfolio(Portfolio.load_json(dict_json["portfolio"]))
        market._event_trans_cost = 0.
        market._mode = None
        if "scenario" in dict_json:
            market.load_scenario(dict_json["scenario"])
        market._agent_markets = dict()
        market._agents = dict()
        market._agent_actions = dict()
        return wrappers.SinglePrecisionWrapper(market)
        
    def jsonify_dict(self) -> dict:
        dict_json = dict()
        dict_json["name"] = self._name
        dict_json["valuation_date"] = self._valuation_date
        dict_json["reward_rule"] = str(self._reward_rule)
        dict_json["hedging_step_in_days"] = self._hedging_step_in_days
        dict_json["num_steps_per_episode"] = self._training_counter.get_total_steps()
        dict_json["validation_rng_seed"] = self._validation_simulator.get_rng_seed()
        dict_json["training_episodes"] = self._training_counter.get_total_paths()
        dict_json["validation_episodes"] = self._validation_counter.get_total_paths()
        dict_json["riskfactorsimulator"] = self._training_simulator.jsonify_dict()
        dict_json["instruments"] = [str(inst) for inst in self._instruments.values()]
        for instrument in self._instruments.values():
            if isinstance(instrument, VarianceSwap):
                varswap_extra_dict = {
                    "instrument": instrument.get_name(),
                    "replicating_opts": instrument.get_hedging_instrument_names(),
                    "excl_realized_var": instrument.get_excl_realized_var()
                }
                if "extra" in dict_json:
                    dict_json["extra"] += [varswap_extra_dict]
                else:
                    dict_json["extra"] = [varswap_extra_dict]
        return dict_json

    def __repr__(self):
        return json.dumps(self.jsonify_dict(), indent=4)
    
    def set_mode(self, mode: str, continue_counter: bool=False):
        """Switch market riskfactor mode
           Training Mode - use _training_simulator
           Validation Mode - use _validation_simulator
           Scenario Mode - use _scenario_simulator
        
        Default training mode is set when constructed
        Args:
            mode (str): training - set to training mode
                        validation - set to validation mode
                        scenario - set to scenario mode
            continue_counter (bool): True - continue counter
                                     False - reset counter
        """
        reset_date()
        if mode == "training":
            self._current_simulator_handler.set_obj(self._training_simulator)
            self._current_counter_handler.set_obj(self._training_counter)
            self._training_simulator.generate_paths(self._training_counter.get_total_paths())
        elif mode == "validation":
            self._current_simulator_handler.set_obj(self._validation_simulator)
            self._current_counter_handler.set_obj(self._validation_counter)
            self._validation_simulator.generate_paths(self._validation_counter.get_total_paths())
        elif mode == "scenario":
            self._current_simulator_handler.set_obj(self._scenario_simulator)
            self._current_counter_handler.set_obj(self._scenario_counter)
        if not continue_counter:
            self._current_counter_handler.get_obj().reset()
        self._mode = mode

    def get_mode(self):
        return self._mode

    def calibrate(self,  
                  underlying: Stock, 
                  listed_options: Union[EuropeanOption, List[List[EuropeanOption]]]=[],
                  model: str="Heston"):
        """Calibrate volatility model by given instruments

        Args:
            underlying (Stock): Underlying stock instrument
            listed_options (Union[EuropeanOption, List[List[EuropeanOption]]]): 
                                European Options used to calibrate the volatility model
                                Heston model: a matrix of European Calls that compose a implied volatility surface
                                BSM:          one European Call (its implied vol will used as GBM's diffusion coefficient)
            model (str): "Heston" or "BSM"
        Raises:
            NotImplementedError: vol_model other than "Heston" or "BSM" is not implemented
        """
        if model == 'BSM':
            param = GBMProcessParam(
                risk_free_rate=self._training_simulator.get_ir(),
                spot=underlying.get_quote(), 
                drift=underlying.get_annual_yield(), 
                dividend=underlying.get_dividend_yield(), 
                vol=listed_options.get_quote(),
                use_risk_free=False
            )
        elif model == 'Heston':
            param = heston_calibration(self._training_simulator.get_ir(), underlying, listed_options)
        else:
            raise NotImplementedError(f'{model} is not supported')
            # param = HestonProcessParam(
            #     risk_free_rate=0.015,
            #     spot=100, 
            #     drift=0.05, 
            #     dividend=0.00,
            #     spot_var=0.096024, kappa=6.288453, theta=0.397888, 
            #     rho=-0.696611, vov=0.753137, use_risk_free=False
            # )
        return param
    
    def get_counter(self):
        return self._current_counter_handler.get_obj()

    def get_instrument(self, instrument_name):
        return self._instruments[instrument_name]

    def add_portfolio(self, portfolio: Portfolio, agent_name: str):
        """set the trading portfolio into market

        Args:
            portfolio (Portfolio): A portfolio that contains a list of exotic instruments (liabilities) 
                                   and a list of eligible hedging instruments
            agent_name (str): agent's portfolio
        """
        agent_market = SingleAgentMarket(self._training_simulator.get_ir(), 
                                         portfolio, self._reward_rule, self._hedging_step_in_days)
        num_steps = max(self._training_counter.get_total_steps(), agent_market.num_steps)
        self._training_counter.set_total_steps(num_steps)
        self._training_simulator.set_num_steps(num_steps)
        # self._training_simulator.generate_paths(self._training_counter.get_total_paths())
        self._validation_counter.set_total_steps(num_steps)
        self._validation_simulator.set_num_steps(num_steps)
        # self._validation_simulator.generate_paths(self._validation_counter.get_total_paths())
        for instrument in self._instruments.values():
            instrument.set_simulator(self._current_simulator_handler, self._current_counter_handler)
        self._agent_markets[agent_name] = wrappers.SinglePrecisionWrapper(agent_market)

    def add_agent(self, agent, agent_name: str):
        self._agents[agent_name] = agent

    def set_trainable_agent(self, agent_name: str):
        self._trainable_agent = agent_name

    def get_agent_market(self, agent_name):
        return self._agent_markets[agent_name]
    
    def get_trainable_agent_market(self):
        return self._agent_markets[self._trainable_agent]

    def get_agent(self, agent_name):
        return self._agents[agent_name]

    def load_scenario(self, scenario_json: Union[List[dict], str]):
        """Load scenario data
        example:
        [
            {
                "name": "AMZN",
                "data": {
                            "time_step_day": 1,
                            "Spot": [3400,3414.063507540342,3360.1097430892696,3514.713081433771,3399.4403346846934,3388.775188349936,3296.0554086124134,3330.74487143777],
                            "Vol 3Mx100": [0.3321,0.3321,0.3321,0.3321,0.3321,0.3321,0.3321,0.3321],
                            "Vol 2Mx100": [0.3321,0.3321,0.3321,0.3321,0.3321,0.3321,0.3321,0.3321],
                            "Vol 4Wx100": [0.3321,0.3321,0.3321,0.3321,0.3321,0.3321,0.3321,0.3321]
                        }
            }
            {
                "name": "SPX",
                "data": {
                            "time_step_day": 1,
                            "Spot": [3400,3414.063507540342,3360.1097430892696,3514.713081433771,3399.4403346846934,3388.775188349936,3296.0554086124134,3330.74487143777],
                            "Vol 3Mx100": [0.3321,0.3321,0.3321,0.3321,0.3321,0.3321,0.3321,0.3321]
                        }
            }
        ]

        Args:
            scenario_json (Union[List[dict], str]): the scenario data in str or list of dict
        """
        self._scenario_simulator.load_json_data(scenario_json)
        self._scenario_counter.set_total_paths(self._scenario_simulator.get_num_paths())
        self._scenario_counter.set_total_steps(self._scenario_simulator.get_num_steps())

    def reset(self):
        """dm_env interface
           Reset episode
        
        Returns:
            [dm_env.TimeStep]: FIRST TimeStep
        """
        reset_date()
        self._current_counter_handler.get_obj().inc_step_counter()
        self._current_counter_handler.get_obj().inc_path_counter() 
        for agent_name, agent_market in self._agent_markets.items():
            if agent_name != self._trainable_agent:
                agent_timestep = agent_market.reset()
                self._agents[agent_name].observe_first(agent_timestep)
                self._agent_actions[agent_name] = self._agents[agent_name].select_action(agent_timestep.observation)
        # with open('logger.csv', 'a') as logger:
        #     logger.write(','.join([str(k) for k in 
        #         [get_cur_days(), 0., 100., 5, 
        #         self._portfolio.get_hedging_portfolio()[0].get_instrument().get_price()[0], 
        #         self._portfolio.get_hedging_portfolio()[0].get_holding(),
        #         self._portfolio.get_liability_portfolio()[0].get_instrument().get_price(), 
        #         self._portfolio.get_liability_portfolio()[0].get_holding(), 
        #         self._cash_account.get_balance(), 
        #         0., '', 0., '']])+'\n'
        #     )
        return self.get_trainable_agent_market().reset()

    def step(self, action):
        """dm_env interface
           step to next time

        Args:
            action (np.darray): hedging buy/sell action from agent

        Returns:
            [dm_env.TimeStep]: MID or LAST TimeStep
        """
        last_period_nav = {}
        trans_costs = {}
        for agent_name, agent_market in self._agent_markets.items():
            # ==============================================================# 
            if agent_name != self._trainable_agent:
                # scale action
                agent_action = self._agent_actions[agent_name]
            else:
                agent_action = action
                self._agent_actions[agent_name] = action
            agent_market.portfolio.scale_actions(agent_action)
            # ==============================================================#
            # Time t                                                        #
            # ==============================================================# 
            
            # ==============================================================#
            # Action                                                        #
            # Take action and rebalance hedging at Time t                   #
            #   Cashflow at Time t                                          #
            #   Transaction Cost at Time t.                                 #
            # ==============================================================#
            cashflow, trans_cost = agent_market.portfolio.rebalance(agent_action) #
            trans_cost += agent_market.event_trans_cost                     #
            trans_costs[agent_name] = trans_cost
            # add any cashflow in/out to funding account                    #
            agent_market.cash_account.add(cashflow)                         #
            # ==============================================================#
            # end of action                                                 #
            # ==============================================================#
            # NAV at time t                                                 #
            last_period_nav[agent_name] = agent_market.portfolio.get_nav()  #   
            
        # adjust cashflow by market impact
        for agent_name, agent_market in self._agent_markets.items():
            agent_action = self._agent_actions[agent_name]
            cashflow = agent_market.portfolio.market_impact(agent_action)
            agent_market.cash_account.add(cashflow)

        reward_extra = dict()
        reward_extra["exceed_constraint"] = self.get_trainable_agent_market() \
                                                .portfolio.get_breach_holding_constraint()
        # ==============================================================#
        # End of Time t                                                 #
        # ==============================================================#

        # ==============================================================#
        # Move to next day                                              #   
        # Time t+1                                                      #
        # ==============================================================#
        move_days(self._hedging_step_in_days)                           #
        self._current_counter_handler.get_obj().inc_step_counter()      #
        # NAV at time t+1    
        trainable_step = None     
        for agent_name, agent_market in self._agent_markets.items():
            agent_action = self._agent_actions[agent_name]
            next_period_nav = agent_market.portfolio.get_nav()                     #
            # Portfolio PnL from Time t => t+1                              #
            # if (np.isnan(next_period_nav)) or (np.isnan(last_period_nav)):  #
            #     portfolio_pnl = LARGE_NEG_VALUE                             #
            # else:                                                           #
            portfolio_pnl = next_period_nav - last_period_nav[agent_name]           #
            # Cash account interest from cash account Time t => t+1         #
            cash_interest = agent_market.cash_account.accrue_interest()            #
            # Event handling, i.e. option exercise                          #
            event_cashflows, agent_market.event_trans_cost = agent_market.portfolio.event_handler()      
            # Add event cash flow into cash account                         #
            agent_market.cash_account.add(event_cashflows)                         #
            # ==============================================================#

            # ==============================================================#
            # Step PnL (Reward) at Time t+1 includes                        #
            #   + portfolio pnl from Time t => t+1                          #
            #   + interest from cash account from Time t => t+1             #
            #   - transaction cost caused by hedging action at Time t+1(t?) #
            # ==============================================================#
            step_pnl = portfolio_pnl + cash_interest - trans_costs[agent_name]           #
            # ==============================================================#
            # print(action, trans_cost)
            # with open('logger.csv', 'a') as logger:
            #     logger.write(','.join([str(k) for k in [get_cur_days(), action[0], self._portfolio.get_hedging_portfolio()[0].get_instrument().get_price()[0], 
            #         self._portfolio.get_hedging_portfolio()[0].get_holding(),
            #         self._portfolio.get_liability_portfolio()[0].get_instrument().get_price(), 
            #         self._portfolio.get_liability_portfolio()[0].get_holding(), 
            #         self._cash_account.get_balance(), 
            #         cash_interest, portfolio_pnl, trans_cost, step_pnl]])+'\n')
            
            # clip action
            agent_market.portfolio.clip_actions(agent_action)
            if self._reach_terminal():
                # Last step at Time T
                if agent_market.portfolio.get_all_liability_expired():
                    # Dump whole portfolio if all positions in liability portfolio expire
                    #   Cashflow at Time T
                    #   Transaction cost at Time T    
                    cashflow, trans_cost = agent_market.portfolio.dump_portfolio()
                    # Step pnl (reward) at Time T also includes
                    #   - Transaction cost caused by dumping the portfolio at Time T
                    step_pnl -= trans_cost
                    # Add the cashflow at Time T into cash account
                    agent_market.cash_account.add(cashflow)
                # print("Cash account Balance: ", self._cash_account.get_balance())
                ret_step = dm_env.termination(
                    reward=agent_market.reward_rule.step_reward(dm_env.StepType.LAST, step_pnl, action, reward_extra),
                    observation=np.append(agent_market._observation(),step_pnl))
            else:
                ret_step = dm_env.transition(
                    reward=agent_market.reward_rule.step_reward(dm_env.StepType.MID, step_pnl, action, reward_extra),
                    observation=np.append(agent_market._observation(),step_pnl),
                    discount=1.)
            if agent_name != self._trainable_agent:
                self._agents[agent_name].observe(agent_action, next_timestep=ret_step)
                observation=np.append(agent_market._observation(),step_pnl)
                self._agent_actions[agent_name] = self._agents[agent_name].select_action(observation)
            else:
                trainable_step = ret_step
        return trainable_step
        
    def _reach_terminal(self) -> bool:
        """ Check if episode reaches terminal step
            If num_steps*hedging_step_in_days equals get_cur_days, 
            then episode reaches terminal step

        Returns:
            bool: True if current time reaches terminal step of the episode
        """
        return ((self._current_counter_handler.get_obj().get_total_steps()) * self._hedging_step_in_days) == get_cur_days()

    def observation_spec(self):
        """dm_env interface
           observation specification returns the observation shapes
           Now the observation includes
                - all hedging positions' price, holding in the portfolio
                - all derivative positions' price, remaining times in the portfolio
                ## - all hedging positions' holding constraint breaching indicator in the portfolio

        Returns:
            [dm_env.specs.Array]: observation specification
        """
        # positions' price, holding; current time; derivatives' remaining times; cash account balance
        
        return self._agent_markets[self._trainable_agent].observation_spec()

    def action_spec(self):
        """Returns the action spec.
        """
        return self._agent_markets[self._trainable_agent].action_spec()

    def __repr__(self):
        return json.dumps(self.jsonify_dict(), indent=4)

    @staticmethod
    def load_market_file(market_file_name):
        market_json = open(market_file_name)
        market_dict = json.load(market_json)
        market_json.close()
        return Market.load_json(market_dict)

    def load_scenario_file(self, scenario_file_name):
        scenario_json = open(scenario_file_name)
        scenario_dict = json.load(scenario_json)
        scenario_json.close()
        self.load_scenario(scenario_dict)

    def get_total_episodes(self):
        return self._current_counter_handler.get_obj().get_total_paths()

    def get_train_episodes(self):
        return self._training_counter.get_total_paths()

    def get_validation_episodes(self):
        return self._validation_counter.get_total_paths()

    def get_scenario_episodes(self):
        return self._scenario_counter.get_total_paths()

if __name__ == "__main__":
    market = Market.load_market_file('Markets/Market_Example/varswap_test1/market.json')
    print(market)
    portfolio = Portfolio.load_portfolio_file('Markets/Market_Example/varswap_test1/portfolio.json')
    print(portfolio)
    market.set_portfolio(portfolio)
    market.set_mode("training")
    stock_prices = np.zeros((100,91))
    call_prices = np.zeros((100,91))
    for i in range(100):
        j=0
        timestep = market.reset()
        stock_prices[i, j] = timestep.observation[0]
        call_prices[i, j] = timestep.observation[4]
        while not timestep.last():
            j += 1
            timestep = market.step([0]*5)
            stock_prices[i, j] = timestep.observation[0]
            call_prices[i, j] = timestep.observation[4]
    import matplotlib.pyplot as plt
    for i in range(100):
        plt.plot(stock_prices[i,:])
    plt.show()
    for i in range(100):
        plt.plot(call_prices[i,:])
    plt.show()
        

    
