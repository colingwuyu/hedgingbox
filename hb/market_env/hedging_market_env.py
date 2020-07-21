import dm_env
import math
from hb.pricing import blackscholes
from typing import List
import numpy as np
from hb.market_env.pathgenerators import pathgenerator
from hb.market_env.rewardrules import reward_rule
from hb.market_env import market_specs


class HedgingMarketEnv(dm_env.Environment):
    """A hedge environment contains European option hedging portfolio.

    """

    def __init__(self,
                 stock_generator: pathgenerator.PathGenerator,
                 reward_rule: reward_rule.RewardRule,
                 market_param: market_specs.MarketEnvParam =
                 market_specs.MarketEnvParam(stock_ticker_size=0.1,
                                             stock_price_upper_bound=np.infty,
                                             stock_price_lower_bound=-np.infty,
                                             lot_size=1,
                                             buy_sell_lots_bound=np.infty,
                                             holding_lots_bound=np.infty),
                 trading_cost_pct: float = 0.01,
                 risk_free_rate: float = 0.,
                 discount_rate: float = 0.,
                 option_maturity: float = 1.,
                 option_strike: float = 100.,
                 option_holding: int = -10,
                 initial_stock_holding: int = 5,
                 obs_attr: List[str] = None):
        """Initializes a new HedgingMarketEnv.

        Args:
            stock_generator (pathgenerator.PathGenerator): Stock price generator
            reward_rule (rewardrule.RewardRule): Reward generator
            market_param (spec.MarketEnvParam): Market specifications of stock and action
            trading_cost_pct (float, optional): Trading cost as percentage of traded value. Defaults to 0.01.
            risk_free_rate (float, optional): Risk free interest rate. Defaults to 0.. It is used for derivate pricing.
            discount_rate (float, optional): Gamma for reward discounting.
            option_maturity (float, optional): Option maturity. Defaults to 1..
            option_strike (float, optional): Option Strike. Defaults to 100..
            option_holding (int, optional): Option Holding. Defaults to -10.
            obs_attr List[str]: A list of observation attributes that env feeds back to actor.
        """
        # intial values for reset
        self._option_maturity = option_maturity
        self._initial_stock_holding = initial_stock_holding
        self._gen_stock_by_step = stock_generator.has_env_interaction()
        # response obs attr
        self._obs_attr = obs_attr
        # market parameters
        self._market_param = market_param
        self._discount_rate = discount_rate
        # stock generator
        self._stock_generator = stock_generator
        self._step_size = self._stock_generator.step_size
        self._num_step = self._stock_generator.num_step
        assert self._num_step * \
            self._step_size <= self._option_maturity, "Stock path longer than option maturity."
        # state space
        self._state_values = {}
        self._state_values['remaining_time'] = option_maturity
        self._state_values['option_price'] = 0.
        self._state_values['option_holding'] = option_holding
        self._state_values['option_strike'] = option_strike
        self._state_values['interest_rate'] = risk_free_rate
        self._state_values['stock_trading_cost_pct'] = trading_cost_pct
        self._state_values['stock_holding'] = initial_stock_holding
        stock_init_price, stock_attr = self._stock_generator.reset_step()
        self._state_values['stock_price'] = stock_init_price
        self._state_values.update(stock_attr)
        # reward generator
        self._reward_rule = reward_rule
        self._reward_rule.reset(self._state_values)
        # gen stock path
        self._stock_path = None
        self._stock_step = None
        if not self._stock_generator.has_env_interaction():
            self._stock_price_path, self._stock_attr_path = \
                self._stock_generator.gen_path(1)
            self._stock_step = 1

    def reset(self):
        """Returns the first `TimeStep` of a new episode.
        """
        # reset state
        self._state_values['stock_holding'] = self._initial_stock_holding
        self._state_values['remaining_time'] = self._option_maturity
        # TODO create a derivative module for pricing
        self._state_values['option_price'] = blackscholes.price(
            call=True, s0=self._state_values['stock_price'],
            r=self._state_values['interest_rate'],
            q=self._state_values['stock_dividend'],
            sigma=self._state_values['stock_sigma'],
            strike=self._state_values['option_strike'],
            tau_e=self._state_values['remaining_time'],
            tau_d=self._state_values['remaining_time']
        )
        stock_init_price, stock_attr = self._stock_generator.reset_step()
        self._state_values['stock_price'] = stock_init_price
        self._state_values.update(stock_attr)
        # gen stock path
        self._stock_price_path = None
        self._stock_attr_path = None
        self._stock_step = 0
        if not self._stock_generator.has_env_interaction():
            self._stock_price_path, self._stock_attr_path = \
                self._stock_generator.gen_path(1)
        # reset reward rule
        self._reward_rule.reset(self._state_values)
        return dm_env.restart(self._observation())

    def step(self, action):
        """Updates the environment according to the action
        """
        buy_sell_action = action[0]
        self._stock_step += 1
        if self._stock_price_path is not None:
            # grab data from path
            stock_price = self._stock_price_path[self._stock_step]
            stock_attr = self._stock_attr_path[self._stock_step]
        else:
            # generate data from generator
            stock_price, stock_attr = \
                self._stock_generator.gen_step(1, self._state_values, action)
        # update state
        self._state_values['stock_price'] = stock_price
        self._state_values.update(stock_attr)
        self._state_values['remaining_time'] -= self._step_size
        self._state_values['option_price'] = blackscholes.price(
            call=True, s0=self._state_values['stock_price'],
            r=self._state_values['interest_rate'],
            q=self._state_values['stock_dividend'],
            sigma=self._state_values['stock_sigma'],
            strike=self._state_values['option_strike'],
            tau_e=self._state_values['remaining_time'],
            tau_d=self._state_values['remaining_time']
        )
        self._state_values['stock_holding'] += buy_sell_action
        if self._stock_step < self._num_step:
            return dm_env.transition(reward=self._reward_rule.step_reward(
                dm_env.StepType.MID,
                self._state_values,
                action),
                observation=self._observation(),
                discount=math.exp(-self._step_size*self._discount_rate))
        else:
            return dm_env.termination(reward=self._reward_rule.step_reward(
                dm_env.StepType.LAST,
                self._state_values, action),
                observation=self._observation())

    def observation_spec(self):
        """Returns the observation spec.
        """
        shape = len(self._obs_attr)
        stock_price_ind = self._obs_attr.index('stock_price')
        holding_ind = self._obs_attr.index('stock_holding')
        t_ind = self._obs_attr.index('remaining_time')
        minimum = [-np.infty]*shape
        maximum = [np.infty]*shape
        discretize_step = np.zeros(shape)
        discretize_step[stock_price_ind] = self._market_param.stock_ticker_size
        discretize_step[holding_ind] = self._market_param.lot_size
        discretize_step[t_ind] = self._step_size*365
        minimum[stock_price_ind] = self._market_param.stock_price_lower_bound
        maximum[stock_price_ind] = self._market_param.stock_price_upper_bound
        minimum[holding_ind] = -self._market_param.holding_lots_bound * \
            self._market_param.lot_size
        maximum[holding_ind] = self._market_param.holding_lots_bound * \
            self._market_param.lot_size
        minimum[t_ind] = (self._option_maturity - self._num_step*self._step_size)*365.
        maximum[t_ind] = self._option_maturity*365.
        return market_specs.DiscretizedBoundedArray(
            shape=(shape,), dtype=float,
            minimum=minimum, maximum=maximum, discretize_step=discretize_step,
            name="market_observations"
        )

    def available_observation_attr(self):
        """Returns available observation attributes
        """
        return self._state_values.keys()

    def get_obs_attr(self) -> List[str]:
        """the env's response observation attributes to actor

        Returns:
            List[str]: a list of observation attributes' names
        """
        return self._obs_attr

    def set_obs_attr(self, obs_attr: List[str]):
        """set the observation attributes for env that are fed back to actor

        Args:
            obs_attr (List[str]): a list of observation attributes' name
        """
        available_attr = self.available_observation_attr()
        for set_attr in obs_attr:
            assert set_attr in available_attr, "%s is not available in environment state space." % set_attr
        self._obs_attr = obs_attr

    def action_spec(self):
        """Returns the action spec.
        """
        minimum = [-self._market_param.lot_size *
                   self._market_param.buy_sell_lots_bound]
        maximum = [self._market_param.lot_size *
                   self._market_param.buy_sell_lots_bound]
        discretize_step = [self._market_param.lot_size]
        return market_specs.DiscretizedBoundedArray(
            shape=(1,), dtype=float,
            minimum=minimum, maximum=maximum, discretize_step=discretize_step,
            name="market_action"
        )

    def _observation(self):
        """Return the observations

        """
        market_observations = np.zeros(len(self._obs_attr), dtype=np.float)
        for ai, obs_attr in enumerate(self._obs_attr):
            market_observations[ai] = self._state_values[obs_attr]
            if obs_attr == 'stock_price':
                # clip to bounds
                market_observations[ai] = max(self._market_param.stock_price_lower_bound, min(self._market_param.stock_price_upper_bound, market_observations[ai]))
            if obs_attr == 'remaining_time':
                # convert to days
                market_observations[ai] = market_observations[ai]*365.
        return market_observations
