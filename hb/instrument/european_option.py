import QuantLib as ql
from hb.instrument.instrument import Instrument
from hb.transaction_cost.transaction_cost import TransactionCost
from hb.utils.process import *
from hb.utils.date import *
from hb.pricing import blackscholes
from hb.utils import consts
import numpy as np
import math
import tf_quant_finance as tff


class EuropeanOption(Instrument):
    def __init__(self, name: str, option_type: str, strike: float, 
                 maturity: float, tradable: bool, quote: float = None,
                 transaction_cost: TransactionCost = None,
                 underlying: Instrument = None, trading_limit: float = 1e10, 
                 reset_time=True):
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if option_type=="Call" else ql.Option.Put, 
                                       strike)
        exercise = ql.EuropeanExercise(add_time(maturity))
        self._option = ql.VanillaOption(payoff, exercise)
        self._maturity_time = maturity
        self._call = option_type=="Call"
        self._strike = strike
        self._back_up_pricing_engine = None
        self._param = None
        self._pred_ind = 0
        super().__init__(name, tradable, quote, transaction_cost, underlying, trading_limit, reset_time=reset_time)

    def get_quote(self) -> float:
        return self._quote
    
    def set_quote(self, quote: float):
        self._quote = quote
    
    def quote(self, quote: float):
        self._quote = quote
        return self

    def get_strike(self) -> float:
        return self._strike

    def get_is_call(self) -> bool:
        return self._call
    
    def get_risk_free_rate(self):
        return self._param.risk_free_rate

    def set_pricing_engine(self, spot_price=None):
        process_param = self._underlying.get_process_param()
        price, underlyer_var = self._underlying.get_price()
        underlyer_price = price
        if spot_price:
            underlyer_price = spot_price
        if isinstance(process_param, HestonProcessParam):
            # Heston model
            self._param = HestonProcessParam(
                    risk_free_rate=process_param.risk_free_rate,spot=underlyer_price, spot_var=max(consts.IMPLIED_VOL_FLOOR, underlyer_var), 
                    drift=process_param.risk_free_rate, dividend=self._underlying.get_dividend_yield(),
                    kappa=process_param.kappa, theta=process_param.theta, 
                    rho=process_param.rho, vov=process_param.vov, use_risk_free=True
                )
            heston_process = create_heston_process(self._param)
            # bsm_process = create_gbm_process(
            #     GBMProcessParam(
            #         risk_free_rate=process_param.risk_free_rate, spot=underlyer_price, 
            #         drift=process_param.risk_free_rate,
            #         dividend=self._underlying.get_dividend_yield(), 
            #         vol=underlyer_var**0.5, use_risk_free=True
            #     )
            # )
            self._option.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(heston_process),0.01,1000))
            self._back_up_pricing_engine = ql.MCEuropeanHestonEngine(
                heston_process, 'pr', timeSteps=days_from_time(self.get_remaining_time()) if self.get_remaining_time() > 0 else 1, requiredSamples=1_000
            )
            # self._back_up_pricing_engine = ql.AnalyticEuropeanEngine(bsm_process)
        elif isinstance(process_param, GBMProcessParam):
            # BSM Model
            self._param = GBMProcessParam(
                    risk_free_rate=process_param.risk_free_rate, spot=underlyer_price, 
                    drift=process_param.risk_free_rate, 
                    dividend=self._underlying.get_dividend_yield(), 
                    vol=process_param.vol, use_risk_free=True
                )
            bsm_process = create_gbm_process(self._param)        
            self._option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))

    def get_is_expired(self) -> bool:
        """check if instrument expires

        Returns:
            bool: True if it expires, False otherwise
        """
        return (self._maturity_time-get_cur_time()) <= 1e-5

    def get_sim_price(self, spot_price=None):
        if (self._cur_price[1] is None):
            # first visit of this timestep, need set up pricing engine
            self.set_pricing_engine()
        if spot_price:
            # use to calculate perturb spot price
            self.set_pricing_engine(spot_price)
        if (abs(self._maturity_time-get_cur_time()) < 1e-5) and (not self._exercised):
            # expiry, not exercised
            option_price = self.get_intrinsic_value()
        elif (self._maturity_time-get_cur_time() < 1e-5) or (self._exercised):
            # past expiry, or exercised
            option_price = 0.
        else:
            # price before expiry
            try:
                price = self._option.NPV()
                # if self._back_up_pricing_engine:
                #     self._option.setPricingEngine(self._back_up_pricing_engine)
                #     with open('logger.csv', 'a') as logger:
                #         mc_price = self._option.NPV()
                #         logger.write(','.join([str(k) for k in [self.get_remaining_time(), price, mc_price, price/mc_price-1]])+'\n')
                # if abs(price -blackscholes.price(True, self._param.spot, self._param.risk_free_rate, 
                #                         self._param.dividend, self._param.vol, self._strike, 
                #                         self.get_remaining_time(), self.get_remaining_time())) > 1e-5:
                #     print(get_date(),
                #         self._param,
                #         price, 
                #         blackscholes.price(True, self._param.spot, self._param.risk_free_rate, 
                #                             self._param.dividend, self._param.vol, self._strike, 
                #                             self.get_remaining_time(), self.get_remaining_time()))
                if price - self.get_intrinsic_value() < -1e-5:
                    raise Exception("Less than intrinsic price")
                option_price = price
            except:
                self._option.setPricingEngine(self._back_up_pricing_engine)
                option_price = self._option.NPV()
        if spot_price:
            # reset pricing engine with current spot
            self.set_pricing_engine()
        return option_price

    def _get_pred_price(self) -> float:
        price = self.get_sim_price()
        return price

    def get_pred_price(self) -> float:
        """Get the prediction price at timestep t
           This function will only be called once at each timestep
           The price will be cached into _cur_price and retrieved directly from get_price() method
        Returns:
            float: prediction price
        """
        if self._cur_pred_file == "Pred_price.csv":
            return self.get_sim_price()
        self.set_pricing_engine()
        self._pred_ind += 1
        if self._cur_pred_path is None:
            self._cur_pred_path = -1
        if (abs(self._cur_price[0]-0.0) < 1e-5):
            # start of a new episode
            if self._cur_pred_path == self._pred_episodes:
                # run out all episodes, start repeating
                self._cur_pred_path = 0
            else:
                # continue next path
                if (self._cur_pred_file is None) and (self._cur_pred_path != -1):
                    self._pred_price_path += [[]]
                self._cur_pred_path += 1
            self._pred_ind = 0
        if self._cur_pred_file:
            # use loaded pred episodes
            price = self._pred_price_path[self._cur_pred_path][self._pred_ind]
        else:
            # simulate pred episodes
            price = self.get_sim_price()
            self._pred_price_path[self._cur_pred_path] = self._pred_price_path[self._cur_pred_path] + [price]
            if (self._pred_ind == self._num_steps) and \
                ((self._cur_pred_path + 1) == self._pred_episodes):
                # save pred episodes:
                self.save_pred_episodes(self._underlying._cur_pred_file)

        return price

    def get_intrinsic_value(self):
        """The discounted intrinsic value

        Returns:
            float: The discounted intrinsic value
        """
        spot, _ = self._underlying.get_price()
        tau_e = self.get_remaining_time()
        r = self._param.risk_free_rate
        q = self._param.dividend
        fwd = spot*math.exp((r-q)*tau_e)
        if self._call:
            intrinsic_value = 0. if fwd <= self._strike else fwd - self._strike
        else:
            intrinsic_value = 0. if fwd >= self._strike else self._strike - fwd
        return math.exp(-r*tau_e)*intrinsic_value

    def get_maturity_time(self):
        return self._maturity_time

    def get_remaining_time(self):
        return self._maturity_time - get_cur_time()

    def get_maturity_date(self):
        return date_from_time(self._maturity_time, ref_date=date0)

    def exercise(self) -> float:
        super().exercise()
        spot, _ = self._underlying.get_price()
        if self._call:
            delivery = 0. if spot <= self._strike else 1.
        else:
            delivery = 0. if spot >= self._strike else -1.
        return delivery

    def get_delivery_amount(self):
        assert abs(self.get_remaining_time()) < 1e-5
        if self._call:
            return 1.
        else:
            return -1.
    
    def get_receive_amount(self):
        assert abs(self.get_remaining_time()) < 1e-5
        if self._call:
            return self._strike
        else:
            return -self._strike

    def get_is_physical_settle(self):
        return True

    def get_implied_vol(self) -> float:
        if isinstance(self._param, GBMProcessParam):
            return self._param.vol
        else:
            return self._implied_vol(
                self._call, self._strike, self.get_remaining_time(),
                self.get_price(),
                GBMProcessParam(
                                risk_free_rate=self._param.risk_free_rate, 
                                spot=self._param.spot,
                                drift=self._param.drift, dividend=self._param.dividend, 
                                vol=0.2,
                                use_risk_free=True
                               )
            )

    def _get_gbm_param(self) -> GBMProcessParam:
        if isinstance(self._param, GBMProcessParam):
            return self._param
        else:
            return GBMProcessParam(
                                risk_free_rate=self._param.risk_free_rate, 
                                spot=self._param.spot,
                                drift=self._param.drift, dividend=self._param.dividend, 
                                vol=self.get_implied_vol(),
                                use_risk_free=True
                            )

    def get_delta(self, use_bs=True) -> float:
        """Delta
            Black-Scholes delta: 
                If underlying follows BSM, then directly use BS delta formula
                If underlying follows Heston, then implied vol is calculated from option price, and then use BS delta formula
            Heston delta:
                Numeric method
                    delta = (Price(S+delta_S) - Price(delta_S)) / delta_S
                    delta_S = 1%*S


        Args:
            use_bs (bool, optional): Use BS Delta or delta from Underlying process. Defaults to True.

        Returns:
            float: [description]
        """
        if (abs(self._maturity_time-get_cur_time()) < 1e-5) and (not self._exercised):
            # expiry, not exercised
            spot, _ = self._underlying.get_price()
            if self._call:
                delta = 0. if spot <= self._strike else 1.
            else:
                delta = 0. if spot >= self._strike else -1.
            return delta
        elif (self._maturity_time-get_cur_time() < 1e-5) or (self._exercised):
            # past expiry, or exercised
            return 0.
        if use_bs or isinstance(self._param, GBMProcessParam):
            # if isinstance(self._param, GBMProcessParam):
            #     delta_value = self._option.delta()
            #     if np.isnan(delta_value):
            #         delta_value = blackscholes.delta_bk(True, self._param.spot, self._param.risk_free_rate, 
            #                                             self._param.dividend, self._strike, 
            #                                             self._param.vol, self.get_remaining_time(), 
            #                                             self.get_remaining_time())
            #     return delta_value
            # else:
            return self._delta(self._call, self._strike,
                                self._get_gbm_param(),
                                self._maturity_time-self._cur_price[0],
                                self._cur_price[1])
        else:
            base_price = self.get_price()
            underlying_price, _ = self._underlying.get_price()
            pertub = underlying_price*0.01
            pertub_spot = underlying_price + pertub
            pertub_price = self.get_sim_price(spot_price=pertub_spot)
            return (pertub_price - base_price) / pertub


    def get_gamma(self) -> float:
        """BS Gamma

        Returns:
            float: BlackScholes Model Gamma
        """
        if (self._maturity_time-get_cur_time() <= -1e-5) or (self._exercised):
            # past expiry / exercised
            return 0.
        if isinstance(self._param, GBMProcessParam):
            return self._option.gamma()
        else:
            return self._gamma(self._call, self._strike,
                               self._get_gbm_param(),
                               self._maturity_time-self._cur_price[0])

    def get_vega(self) -> float:
        """BS Vega

        Returns:
            float: BlackScholes Model Vega
        """
        if (self._maturity_time-get_cur_time() <= -1e-5) or (self._exercised):
            # past expiry / exercised
            return 0.
        if isinstance(self._param, GBMProcessParam):
            return self._option.vega()
        else:
            return self._vega(self._call, self._strike,
                              self._get_gbm_param(),
                              self._maturity_time-self._cur_price[0])
    @classmethod
    def _implied_vol(cls, call, strike, tau_e, option_price, gbm_param):
        euro_opt = cls("impl_vol", 'Call' if call else 'Put', strike,
                       tau_e, False, reset_time=False)
        process = create_process(gbm_param)
        try:
            return euro_opt._option.impliedVolatility(option_price, process)
        except:
            return consts.IMPLIED_VOL_FLOOR

    @classmethod
    def _create_euro_opt(cls, call, strike, param, tau_e, price=None):
        if price:
            sigma = cls._implied_vol(call, strike, tau_e, price, 
                                    GBMProcessParam(
                                        risk_free_rate=param.risk_free_rate, spot=param.spot,
                                        drift=param.drift, dividend=param.dividend, vol=0.2,
                                        use_risk_free=True
                                    ))
            bsm_process = create_process(
                                GBMProcessParam(
                                    risk_free_rate=param.risk_free_rate, spot=param.spot,
                                    drift=param.drift, dividend=param.dividend, vol=sigma,
                                    use_risk_free=True
                                ))
        else:
            assert isinstance(param, GBMProcessParam)
            bsm_process = create_process(param)
        euro_opt = EuropeanOption("impl_vol", 'Call' if call else 'Put', strike,
                                tau_e, False, reset_time=False)
        euro_opt._option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
        return euro_opt

    @classmethod
    def _delta(cls, call, strike, param, tau_e, price=None):
        delta_value = cls._create_euro_opt(call, strike, param, tau_e, price)._option.delta()
        if np.isnan(delta_value):
            delta_value = blackscholes.delta_bk(True, param.spot, param.risk_free_rate, param.dividend, strike, param.vol, tau_e, tau_e)
        return delta_value
    
    @classmethod
    def _gamma(cls, call, strike, param, tau_e, price=None):
        return cls._create_euro_opt(call, strike, param, tau_e, price)._option.gamma()

    @classmethod
    def _vega(cls, call, strike, param, tau_e, price=None):
        return cls._create_euro_opt(call, strike, param, tau_e, price)._option.vega()


    def __repr__(self):
        return f'European Option {self._name}: \nunderlying=({str(self._underlying)})\noption_type={self._call}, strike={self._strike}, maturity={get_period_str_from_time(self._maturity_time)}, tradable={self._tradable}, iv={self._quote}, transaction_cost={str(self._transaction_cost)}'


if __name__ == "__main__":
    from hb.transaction_cost.percentage_transaction_cost import PercentageTransactionCost
    from hb.instrument.instrument_factory import InstrumentFactory
    from hb.utils.date import *
    from hb.utils.consts import *
    from hb.utils.process import *
    import numpy as np
    
    spx = InstrumentFactory.create(
        'Stock AMZN 3400 25 0 0.15'
    )
    print(spx)
    spx_3m = InstrumentFactory.create(
        'EuroOpt AMZN Listed 3M Call 3400 49.25 5 (AMZN_ATM_3M_CALL)'
    ).underlying(spx)
    print(spx_3m)
    risk_free_rate = 0.015
    heston_param = HestonProcessParam(
            risk_free_rate=0.015,
            spot=spx.get_quote(), 
            drift=spx.get_annual_yield(), 
            dividend=spx.get_dividend_yield(),
            spot_var=0.096024, kappa=6.288453, theta=0.397888, 
            rho=-0.696611, vov=0.753137, use_risk_free=False
        )
    gbm_param = GBMProcessParam(
            risk_free_rate = 0.015,
            spot=spx.get_quote(), 
            drift=spx.get_annual_yield(), 
            dividend=spx.get_dividend_yield(), 
            vol=0.5,
            use_risk_free=False
        )
    
    num_path = 1_000
    num_step = 90    
    step_days = 1
    step_size = time_from_days(step_days)
    spx.set_pricing_engine(step_size, num_step, heston_param)
    heston_prices = np.zeros([num_path, num_step])
    times = np.zeros([num_step])
    
    for i in range(num_path):
        for j in range(num_step):
            times[j] = get_cur_days()
            # print("Days ", get_cur_days())
            heston_prices[i][j] = spx_3m.get_price()
            move_days(step_days)
        reset_date()
        
    spx.set_pricing_engine(step_size, num_step, gbm_param)
    gbm_prices = np.zeros([num_path, num_step])
    for i in range(num_path):
        for j in range(num_step):
            # print(get_cur_days())
            gbm_prices[i][j] = spx_3m.get_price()
            move_days(step_days)
        reset_date()

    import matplotlib.pyplot as plt

    for i in range(num_path):
        plt.plot(times, heston_prices[i, :], lw=0.8, alpha=0.6)
    plt.title("Heston Simulation Option")
    plt.show()
    print(heston_prices[:,-1].mean())
    for i in range(num_path):
        plt.plot(times, gbm_prices[i, :], lw=0.8, alpha=0.6)
    plt.title("GBM Simulation")
    plt.show()
    print(gbm_prices[:,-1].mean())
