from hb.instrument.instrument import Instrument
from hb.instrument.stock import Stock
from hb.instrument.european_option import EuropeanOption
from hb.instrument.cash_account import CashAccount
from hb.market_env.rewardrules.pnl_reward import PnLReward
from hb.market_env import market_specs
from hb.market_env.rewardrules import reward_rule
from hb.utils.date import *
from hb.utils.process import *
from hb.utils.heston_calibration import *
from hb.market_env.portfolio import Portfolio
import math
from typing import List, Union
import numpy as np
import dm_env
from dm_env import specs
from os.path import join as pjoin


class Market(dm_env.Environment):
    """Market Environment
    """
    def __init__(
            self,
            reward_rule: reward_rule.RewardRule=PnLReward(),  
            risk_free_rate: float=0.01,
            hedging_step_in_days: int=1,
            vol_model: str = 'BSM', 
            name: str = 'Market',
            dir_: str = 'Market'
        ):
        reset_date()
        self._risk_free_rate = risk_free_rate
        self._underlying_processes = dict()
        self._underlying_processes_param = dict()
        self._tradable_products = np.array([])
        self._tradable_products_map = dict()
        self._exotic_products = np.array([])
        self._exotic_products_map = dict()
        self._cash_account = CashAccount(interest_rates=risk_free_rate)
        self._pnl_reward = PnLReward()
        self._reward_rule = reward_rule
        self._hedging_step_in_days = hedging_step_in_days
        self._num_steps = 0
        self._portfolio = None
        self._pred_episodes = None
        self._vol_model = vol_model
        self._name = name
        self._event_trans_cost = 0.
        self._dir = pjoin(dir_, "Market_" + name + "_VolModel_" + vol_model)

    def get_dir(self):
        return self._dir
    
    def set_pred_mode(self, pred_mode: bool):
        """Set market to prediction mode
           Prediction Mode will generate a fixed pred_episodes

        Args:
            pred_mode (bool): True - set to prediction mode
                              False - set to non-prediction mode
        """
        for position in self._portfolio.get_portfolio_positions():
            position.get_instrument().set_pred_mode(pred_mode)

    def get_num_steps(self):
        return self._num_steps

    def set_num_steps(self, num_steps: int):
        self._num_steps = num_steps
        for position in self._portfolio.get_portfolio_positions():
            position.get_instrument().set_num_steps(self._num_steps)

    def get_pred_episodes(self):
        return self._pred_episodes

    def set_pred_episodes(self, pred_episodes):
        self._pred_episodes = pred_episodes
        for instrument in self._tradable_products:
            instrument.set_pred_episodes(pred_episodes)
        for instrument in self._exotic_products:
            instrument.set_pred_episodes(pred_episodes)

    def add_instrument(self, instrument: Instrument):
        """add instrument into market, which will be available to construct portfolio

        Args:
            instrument (Instrument): Instrument add to market
        """
        if isinstance(instrument, Stock):
            instrument.set_pred_episodes(self._pred_episodes)
        if instrument.get_is_tradable():
            self._tradable_products = np.append(self._tradable_products, instrument)
            self._tradable_products_map[instrument.get_name()] = instrument
        else:
            self._exotic_products = np.append(self._exotic_products, instrument)
            self._exotic_products_map[instrument.get_name()] = instrument

    def add_instruments(self, instruments: List[Instrument]):
        """add a list of instruments into market, which will be available to construct portfolio

        Args:
            instruments (List[Instrument]): A list of instruments add to market
        """
        for i in instruments:
            self.add_instrument(i)
    
    def set_vol_model(self, vol_model: str):
        """vol_model setter

        Args:
            vol_model (str): Volatility model - "Heston", or "BSM"
        """
        self._vol_model = vol_model
    
    def get_vol_model(self):
        return self._vol_model

    def calibrate(self,  
                  underlying: Stock, 
                  listed_options: Union[EuropeanOption, List[List[EuropeanOption]]]):
        """Calibrate volatility model by given instruments

        Args:
            underlying (Stock): Underlying stock instrument
            listed_options (Union[EuropeanOption, List[List[EuropeanOption]]]): 
                                European Options used to calibrate the volatility model
                                Heston model: a matrix of European Calls that compose a implied volatility surface
                                BSM:          one European Call (its implied vol will used as GBM's diffusion coefficient)

        Raises:
            NotImplementedError: vol_model other than "Heston" or "BSM" is not implemented
        """
        if self._vol_model == 'BSM':
            param = GBMProcessParam(
                risk_free_rate=self._risk_free_rate,
                spot=underlying.get_quote(), 
                drift=underlying.get_annual_yield(), 
                dividend=underlying.get_dividend_yield(), 
                vol=listed_options.get_quote(),
                use_risk_free=False
            )
        elif self._vol_model == 'Heston':
            param = heston_calibration(self._risk_free_rate, underlying, listed_options)
        else:
            raise NotImplementedError(f'{vol_model} is not supported')
        # param = HestonProcessParam(
        #     risk_free_rate=0.015,
        #     spot=100, 
        #     drift=0.05, 
        #     dividend=0.00,
        #     spot_var=0.096024, kappa=6.288453, theta=0.397888, 
        #     rho=-0.696611, vov=0.753137, use_risk_free=False
        # )
        underlying.set_process_param(param)
        self.add_instrument(underlying)
        self.add_instruments(np.array(listed_options).flatten())

    def get_instrument(self, instrument_name: str) -> Instrument:
        """Get instrument from the market

        Args:
            instrument_name (str): name of the instrument

        Returns:
            Instrument: Instrument object
        """
        if instrument_name in self._tradable_products_map:
            return self._tradable_products_map[instrument_name]
        else:
            return self._exotic_products_map[instrument_name]

    def get_instruments(self, instrument_names: List[str]) -> List[Instrument]:
        """Get a list of instruments from the market

        Args:
            instrument_names (List[str]): a list of instrument names

        Returns:
            List[Instrument]: a list of Instrument objects
        """
        return [self.get_instrument(i) for i in instrument_names]

    def init_portfolio(self, portfolio: Portfolio):
        """Initialize the trading portfolio in the market

        Args:
            portfolio (Portfolio): A portfolio that contains a list of exotic instruments (liabilities) 
                                   and a list of eligible hedging instruments
        """
        self._num_steps = 0
        for derivative in portfolio.get_liability_portfolio():
            num_steps_to_maturity = int(days_from_time(derivative.get_instrument().get_maturity_time()) / self._hedging_step_in_days)
            self._num_steps = max(self._num_steps, num_steps_to_maturity)
        self._portfolio = portfolio 
        self.set_num_steps(self._num_steps)
        for position in portfolio.get_portfolio_positions():
            if isinstance(position.get_instrument(), Stock):
                # set up Stock evolving process by step size 
                position.get_instrument().set_pricing_engine(time_from_days(self._hedging_step_in_days))
        self._portfolio.set_market_dir(self._dir)
        
    def load_scenario(self, scenario_name: str):
        """Load scenario path of instruments to the market

        Args:
            scenario_name (str): the scenario file name of instruments
        """
        for position in self._portfolio.get_portfolio_positions():
            instrument = position.get_instrument()
            if isinstance(instrument, Stock):
                self._pred_episodes, self._num_steps = \
                    instrument.load_pred_episodes(pred_file=scenario_name+'_price.csv', 
                                                  pred_vol_file=scenario_name+'_vol.csv')
            else:
                instrument.load_pred_episodes(pred_file=scenario_name+'_price.csv')
        self.set_pred_mode(True)
        self.set_pred_episodes(self._pred_episodes)
        self.set_num_steps(self._num_steps)

    def reset(self):
        """dm_env interface
           Reset episode
        
        Returns:
            [dm_env.TimeStep]: FIRST TimeStep
        """
        reset_date()
        self._portfolio.reset()
        self._cash_account.reset()
        initial_cashflow = self._portfolio.get_nav()
        # save initial cashflow into cash account
        self._cash_account.add(-initial_cashflow)
        self._reward_rule.reset(self._portfolio)
        self._pnl_reward.reset(self._portfolio)
        return dm_env.restart(np.append(self._observation(), 0.))

    def step(self, action):
        """dm_env interface
           step to next time

        Args:
            action (np.darray): hedging buy/sell action from agent

        Returns:
            [dm_env.TimeStep]: MID or LAST TimeStep
        """
        # ==============================================
        # Time t
        # take action and rebalance hedging at Time t
        #   Cashflow at Time t
        #   Transaction Cost at Time t
        cashflow, trans_cost = self._portfolio.rebalance(action)
        trans_cost += self._event_trans_cost
        # add any cashflow in/out to funding account
        self._cash_account.add(cashflow)
        last_period_nav = self._portfolio.get_nav()
        # ==============================================
        # move to next day
        # Time t+1
        move_days(self._hedging_step_in_days)
        next_period_nav = self._portfolio.get_nav()
        # portfolio pnl from Time t => t+1
        portfolio_pnl = next_period_nav - last_period_nav
        # cash interest from cash account Time t => t+1
        cash_interest = self._cash_account.accrue_interest()
        # event handling, i.e. option exercise 
        event_cashflows, self._event_trans_cost = self._portfolio.event_handler()
        self._cash_account.add(event_cashflows)
        # step pnl (reward) at Time t+1 includes 
        #   + portfolio pnl from Time t => t+1
        #   + interest from cash account from Time t => t+1 
        #   - transaction cost caused by hedging action at Time t
        step_pnl = portfolio_pnl + cash_interest - trans_cost
        # print(action, trans_cost)
        # with open('logger.csv', 'a') as logger:
        #     logger.write(','.join([str(k) for k in [get_cur_days(), action[0], self._portfolio.get_hedging_portfolio()[0].get_instrument().get_price()[0], 
        #         self._portfolio.get_hedging_portfolio()[0].get_holding(),
        #         self._portfolio.get_liability_portfolio()[0].get_instrument().get_price(), 
        #         self._portfolio.get_liability_portfolio()[0].get_holding(), 
        #         self._cash_account.get_balance(), 
        #         cash_interest, portfolio_pnl, trans_cost, step_pnl]])+'\n')
        if self._reach_terminal():
            # last step at Time T
            # dump whole portfolio
            #   Cashflow at Time T
            #   Transaction cost at Time T
            cashflow, trans_cost = self._portfolio.dump_portfolio()
            # step pnl (reward) at Time T also includes
            #   - transaction cost caused by dumping the portfolio at Time T
            step_pnl -= trans_cost
            # add the cashflow at Time T into cash account
            self._cash_account.add(cashflow)
            # print("Cash account Balance: ", self._cash_account.get_balance())
            ret_step = dm_env.termination(
                reward=self._reward_rule.step_reward(dm_env.StepType.LAST, step_pnl),
                observation=np.append(self._observation(),
                                      self._pnl_reward.step_reward(dm_env.StepType.LAST, step_pnl)))
        else:
            ret_step = dm_env.transition(
                reward=self._reward_rule.step_reward(dm_env.StepType.MID, step_pnl),
                observation=np.append(self._observation(),
                                      self._pnl_reward.step_reward(dm_env.StepType.MID, step_pnl)),
                discount=0.)
        return ret_step
        
    def _reach_terminal(self) -> bool:
        """ Check if episode reaches terminal step
            If num_steps*hedging_step_in_days equals get_cur_days, 
            then episode reaches terminal step

        Returns:
            bool: True if current time reaches terminal step of the episode
        """
        return (self._num_steps * self._hedging_step_in_days) == get_cur_days()

    def _observation(self):
        """Construct state observation
           Now the observation includes
                - all hedging positions' price, holding in the portfolio
                - all derivative positions' price, remaining times in the portfolio

        Returns:
            market_observations [np.darray]: a list of state observation
        """
        market_observations = np.array([], dtype=np.float)
        for position in self._portfolio.get_hedging_portfolio():
            if isinstance(position.get_instrument(), Stock):
                price, _ = position.get_instrument().get_price()
            else:
                price = position.get_instrument().get_price()
            # add position's price and holding
            market_observations = np.append(market_observations, [price, position.get_holding()])
        for position in self._portfolio.get_liability_portfolio():
            market_observations = np.append(market_observations, [position.get_instrument().get_price(), 
                                                                  position.get_instrument().get_remaining_time()])
        return market_observations
    
    def observation_spec(self):
        """dm_env interface
           observation specification returns the observation shapes
           Now the observation includes
                - all hedging positions' price, holding in the portfolio
                - all derivative positions' price, remaining times in the portfolio

        Returns:
            [dm_env.specs.Array]: observation specification
        """
        # positions' price, holding; current time; derivatives' remaining times; cash account balance
        num_obs = 2*len(self._portfolio.get_portfolio_positions())
        obs_shape = (num_obs, )
        return specs.Array(
            shape=obs_shape, dtype=float, name="market_observations"
        )

    def action_spec(self):
        """Returns the action spec.
        """
        maximum = []
        for hedging_position in self._portfolio.get_hedging_portfolio():
            maximum += [hedging_position.get_instrument().get_trading_limit()]
        maximum = np.array(maximum)
        minimum = -1*maximum
        return specs.BoundedArray(
            shape=(len(maximum),), dtype=float,
            minimum=minimum, maximum=maximum, name="hedging_actions"
        )

    def __repr__(self):
        market_str = 'Market Information: \n'
        market_str += '     Risk Free Rate: \n'
        market_str += f'        {self._risk_free_rate}\n'
        market_str += '     Hedging Step in Days: \n'
        market_str += f'        {self._hedging_step_in_days}\n'
        market_str += '     Reward Rule: \n'
        market_str += f'        {type(self._reward_rule)}\n'
        market_str += f'    Instruments: \n'
        market_str += '         Tradable:\n'
        for i in self._tradable_products:
            market_str += f'            {str(i)}\n'
        market_str += '         Exotic:\n'
        for i in self._exotic_products:
            market_str += f'            {str(i)}\n'
        market_str += f'    Portfolio: \n'
        if self._portfolio is None:
            market_str += '         Not initialized'
        else:
            for position in self._portfolio.get_portfolio_positions():
                market_str += f'         {position.get_instrument().get_name()} Holding {position.get_holding()}\n'
        return market_str

    
