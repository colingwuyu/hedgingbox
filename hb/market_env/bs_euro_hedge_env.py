import dm_env
from dm_env import specs
from hb.pricing import blackscholes
import numpy as np


class BSEuroHedgeEnv(dm_env.Environment):
    """A hedge environment contains European option hedging portfolio.

    """

    def __init__(self,
                 episode_steps: int = 365,
                 trading_cost_pct: float = 0.01,
                 interest_rate: float = 0.,
                 option_maturity: float = 1.,
                 option_strike: float = 100.,
                 option_holding: int = -10,
                 stock_init_price: float = 100.,
                 stock_drift: float = 0.05,
                 stock_dividend: float = 0.,
                 stock_sigma: float = 0.2,
                 max_buy_action: float = 5.,
                 max_sell_action: float = -5.,
                 seed: int = 1):
        """Initializes a new BSEuroHedgeEnv.

        Args:
            seed (int, optional): random seed for the RNG. Defaults to 1.
        """
        self._rng = np.random.RandomState(seed)
        self._step_size = option_maturity / episode_steps
        # intial values for reset
        self._option_maturity = option_maturity
        self._stock_init_price = stock_init_price
        # state observations
        # self._remaining_time = option_maturity
        # self._option_holding = option_holding
        # self._option_strike = option_strike
        # self._interest_rate = interest_rate
        # self._stock_trading_cost_pct = trading_cost_pct
        # self._stock_price = stock_init_price
        # self._stock_drift = stock_drift
        # self._stock_dividend = stock_dividend
        # self._stock_sigma = stock_sigma
        # self._stock_holding = 0.

        # action space
        self._max_sell = max_sell_action
        self._max_buy = max_buy_action
        # state space
        self._state_values = {}
        self._state_values['remaining_time'] = option_maturity
        self._state_values['option_price'] = 0.
        self._state_values['option_holding'] = option_holding
        self._state_values['option_strike'] = option_strike
        self._state_values['interest_rate'] = interest_rate
        self._state_values['stock_trading_cost_pct'] = trading_cost_pct
        self._state_values['stock_price'] = stock_init_price
        self._state_values['stock_drift'] = stock_drift
        self._state_values['stock_dividend'] = stock_dividend
        self._state_values['stock_sigma'] = stock_sigma
        self._state_values['stock_holding'] = 0.
        self._prev_state_values = self._state_values.copy()
        # state
        #   - option value
        #   - stock price
        #   - stock holding
        #   - time to maturity
        # self._state = np.zeros(4, dtype=np.float)

    def reset(self):
        """Returns the first `TimeStep` of a new episode.
        """
        self._state_values['stock_price'] = self._stock_init_price
        self._state_values['stock_holding'] = 0.
        self._state_values['remaining_time'] = self._option_maturity
        self._state_values['option_price'] = blackscholes.price(
            call=True, s0=self._state_values['stock_price'],
            r=self._state_values['interest_rate'],
            q=self._state_values['stock_dividend'],
            sigma=self._state_values['stock_sigma'],
            strike=self._state_values['option_strike'],
            tau_e=self._state_values['remaining_time'],
            tau_d=self._state_values['remaining_time']
        )
        return dm_env.restart(self._observation())

    def step(self, action):
        """Updates the environment according to the action
        """
        buy_sell_action = action[0]
        self._prev_state_values = self._state_values.copy()
        self._state_values['stock_price'] += self._state_values['stock_price'] * \
            ((self._state_values['stock_drift'] - self._state_values['stock_dividend']) *
             self._step_size + self._state_values['stock_sigma'] *
             np.sqrt(self._step_size)*self._rng.normal(0, 1, 1)[0])
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
        if self._state_values['remaining_time'] >= 0.:
            if self._prev_state_values['stock_holding'] == 0.:
                # r_0 = -k|S_0*H_0|
                pnl = -self._state_values['stock_trading_cost_pct'] * \
                    self._state_values['stock_price']*abs(buy_sell_action)
            else:
                # r_i = V_i - V_{i-1} + H_{i-1}(S_i - S_{i-1}) - k|S_i*(H_i-H_{i-1})|
                # A_i = H_i - H_{i-1}
                pnl = (self._state_values['option_price'] - self._prev_state_values['option_price']) * \
                    self._state_values['option_holding'] \
                    + self._prev_state_values['stock_holding'] * \
                      (self._state_values['stock_price'] - self._prev_state_values['stock_price']) \
                    - self._state_values['stock_trading_cost_pct'] * \
                    self._state_values['stock_price']*abs(buy_sell_action)
            return dm_env.transition(reward=pnl,
                                     observation=self._observation())
        else:
            # end of episode
            # liquidate all stocks
            # r_i = -k|S_i*H_i|
            pnl = -self._state_values['stock_trading_cost_pct'] * \
                self._state_values['stock_price'] * \
                self._state_values['stock_holding']
            return dm_env.termination(reward=pnl,
                                      observation=self._observation())

    def observation_spec(self):
        """Returns the observation spec.
        """
        return specs.Array(
            shape=(11,), dtype=np.float,
            name="market_observations"
        )

    def action_spec(self):
        """Returns the action spec.
        """
        return specs.BoundedArray(
            shape=(1,), dtype=float,
            minimum=[self._max_sell], maximum=[self._max_buy],
            name="buy_sell_action"
        )

    def _observation(self):
        """Return the observations

        """
        market_observations = np.zeros(11, dtype=np.float)
        market_observations[0] = self._state_values['remaining_time']
        market_observations[1] = self._state_values['option_price']
        market_observations[2] = self._state_values['option_holding']
        market_observations[3] = self._state_values['option_strike']
        market_observations[4] = self._state_values['interest_rate']
        market_observations[5] = self._state_values['stock_trading_cost_pct']
        market_observations[6] = self._state_values['stock_price']
        market_observations[7] = self._state_values['stock_drift']
        market_observations[8] = self._state_values['stock_dividend']
        market_observations[9] = self._state_values['stock_sigma']
        market_observations[10] = self._state_values['stock_holding']
        return market_observations
