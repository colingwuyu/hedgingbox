from typing import Union
import json
from hb.market_env.market import Market
from hb.utils.loggers.default import make_default_logger
import hb.bots.d4pgbot as d4pg
import hb.bots.greekbot as greek
import acme
from acme.utils import counting
import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt


class Preset:
    def __init__(self) -> None:
        self._market = None
        self._agent = None
        self._agent_type = None
        self._log_path = None
        self._loop = None

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
            }
        }
        Args:
            json_ (Union[dict, str]): market environment in json
        """
        if isinstance(json_, str):
            dict_json = json.loads(json_)
        else:
            dict_json = json_
        preset = cls()
        market = Market.load_json(dict_json["market"])
        preset._log_path = dict_json["log_path"]
        if not os.path.exists(preset._log_path):
            os.makedirs(preset._log_path)
        for agent in dict_json["agents"]:
            if agent["agent_type"] == "D4PG":
                if agent["name"] == dict_json["trainable_agent"]:
                    agent_obj = d4pg.load_json(
                        agent, market, preset._log_path, True)
                    preset._reward_func = lambda stats: agent["parameters"]["mu_lambda"] * \
                        stats[0] - agent["parameters"]["risk_obj_c"]*stats[1]
                else:
                    agent_obj = d4pg.load_json(
                        agent, market, preset._log_path, False)
            elif agent["agent_type"] == "Greek":
                agent_obj = greek.load_json(agent, market, preset._log_path)
            else:
                raise NotImplementedError()
            market.add_agent(agent_obj, agent["name"])
            if agent["name"] == dict_json["trainable_agent"]:
                market.set_trainable_agent(agent["name"])
                preset._agent = market.get_agent(dict_json["trainable_agent"])
                preset._agent_type = agent["agent_type"]
        preset._market = market
        preset._environment_log_file = os.path.join(
            preset._log_path, "logs/train_loop/logs.csv")
        counter = counting.Counter()
        preset._best_reward = None
        if preset._agent_type == 'D4PG':
            if os.path.exists(preset._environment_log_file):
                loop_log = pd.read_csv(preset._environment_log_file)
                counter.increment(
                    episodes=loop_log.episodes.values[-1], steps=loop_log.steps.values[-1])
                num_train_episodes = market.get_train_episodes()
                for i in range(int(len(loop_log)/num_train_episodes)):
                    last_set = loop_log.episode_return.values[(
                        num_train_episodes*i):(num_train_episodes*(i+1))]
                    last_mean = np.mean(last_set)
                    last_std = np.std(last_set)
                    last_reward = preset._reward_func([last_mean, last_std])
                    if (preset._best_reward is None) or (last_reward > preset._best_reward):
                        preset._best_reward = last_reward
        print("Current best target function value is: ", preset._best_reward)
        preset._loop = acme.EnvironmentLoop(preset._market, preset._agent,
                                            logger=make_default_logger(
                                                directory=preset._log_path,
                                                label="train_loop",
                                                time_delta=0.0),
                                            counter=counter)
        validation_log_path = os.path.join(preset._log_path, "logs/validation_loop")                                   
        if os.path.exists(validation_log_path):
            shutil.rmtree(validation_log_path)
        preset._validation_loop = acme.EnvironmentLoop(preset._market, preset._agent,
                                                       logger=make_default_logger(
                                                           directory=preset._log_path,
                                                           label="validation_loop",
                                                           time_delta=0.0))

        return preset

    @staticmethod
    def load_file(preset_file):
        preset_json = open(preset_file)
        preset_dict = json.load(preset_json)
        preset_json.close()
        return Preset.load_json(preset_dict)

    def dist_stat_save(self):
        for agent in self._market._agents.values():
            predictor = agent.get_predictor()
            bot_name = agent.get_name()
            predictor._update_progress_figures()
            status = predictor._progress_measures
            print(f"{bot_name} Bot PnL mean-var %s" % str(status['mean-var']))
            print(f"{bot_name} Bot PnL mean %s" % str(status['pnl_mean']))
            print(f"{bot_name} Bot PnL std %s" % str(status['pnl_std']))
            print(f"{bot_name} Bot 95VaR %s" % status['pnl_95VaR'])
            print(f"{bot_name} Bot 99VaR %s" % status['pnl_99VaR'])
            print(f"{bot_name} Bot 95CVaR %s" % status['pnl_95CVaR'])
            print(f"{bot_name} Bot 99CVaR %s" % status['pnl_99CVaR'])
            status_dic = {}

            for k in status.keys():
                status_dic[k] = [status[k]]
            log_path = os.path.dirname(
                agent.get_predictor().get_perf_log_file_path())
            pd.DataFrame.from_dict(status_dic, orient="index", columns=[bot_name]).to_csv(
                f'{log_path}/{bot_name}_pnl_stat.csv')

    def train(self, num_check_points):
        if self._agent_type == "Greek":
            print("Greek agent is not trainable.")
            return
        else:
            self._agent.set_pred_only(False)

            # Try running the environment loop. We have no assertions here because all
            # we care about is that the agent runs without raising any errors.

            # num_prediction_episodes = self._market.get_validation_episodes()
            num_prediction_episodes = 0
            num_train_episodes = self._market.get_train_episodes()
            num_episodes = num_train_episodes + num_prediction_episodes
            if num_episodes > 0:
                for i in range(num_check_points):
                    print(f"Check Point {i}")
                    # train
                    self._market.set_mode("training", continue_counter=True)
                    self._loop.run(num_episodes=num_train_episodes)
                    # prediction
                    # self._market.set_mode("validation")
                    # self._loop.run(num_episodes=num_prediction_episodes)
                    # if self._agent.get_predictor().is_best_perf():
                    loop_log = pd.read_csv(self._environment_log_file)
                    last_set = loop_log.episode_return.values[(
                        -num_train_episodes):]
                    last_mean = np.mean(last_set)
                    last_std = np.std(last_set)
                    last_reward = self._reward_func([last_mean, last_std])
                    print("Checkpoint target function value is: ", last_reward)
                    if (self._best_reward is None) or (last_reward > self._best_reward):
                        print("Best target function value, saving model...")
                        self._best_reward = last_reward
                        self._agent.checkpoint_save()

    def validation(self):
        self._market.set_mode("validation")
        for agent in self._market._agents.values():
            agent.set_pred_only(True)
        self._validation_loop.run(num_episodes=self._market.get_validation_episodes())
        self.dist_stat_save()

        for agent in self._market._agents.values():
            hedge_perf = pd.read_csv(
                agent.get_predictor().get_perf_log_file_path())
            hedge_pnl_list = hedge_perf[hedge_perf.type == 'pnl'].drop(
                ['path_num', 'type'], axis=1).sum(axis=1).values
            log_path = os.path.dirname(
                agent.get_predictor().get_perf_log_file_path())
            np.save(
                f'{log_path}/{agent.get_name()}hedge_pnl_measures.npy', hedge_pnl_list)

    def plot_progress(self, start_episode=0):
        train_progress = pd.read_csv(
            self._loop._logger._to._to._to[1]._file_path)
        plt.plot(train_progress.episodes[start_episode:],
                 train_progress.episode_return[start_episode:])

    def plot_pnl_distribution(self):
        for agent in self._market._agents.values():
            log_path = os.path.dirname(
                agent.get_predictor().get_perf_log_file_path())
            pnl_path = f'{log_path}/{agent.get_name()}hedge_pnl_measures.npy'
            hedge_pnl_list = np.load(pnl_path)
            plt.hist(hedge_pnl_list, bins=50, alpha=0.5,
                     label=agent.get_name(), density=True)
        plt.legend(loc='upper left')
        plt.show()

    # def plot_path(self, pnl, action, cum_holding,
    #           hedging_inst, hedging_price, liability_inst, derivative_price,
    #           figure_name):
    #     logs_file = {
    #         "d4pg hedging": 'logs/d4pg_predictor/performance/logs.csv',
    #         "greek hedging": 'logs/greek_delta_hedging/performance/logs.csv',
    #         # "d4pg hedging": 'logs/d4pg_predictor_bmk/performance/logs.csv'
    #     }
    #     hedging_inst = ["BAC", "XLF"]
    #     liability_inst = ["BAC Call"]

    #     perfs = {}
    #     for n, f in logs_file.items():
    #         perfs[n] = pd.read_csv(model_path + f)
    #     hedging_strategy = "d4pg hedging"
    #     price_from = "d4pg hedging"
    #     all_pnl_list = {}
    #     for n, perf in perfs.items():
    #         all_pnl_list[n] = perf[perf.type=='pnl'].drop(['path_num','type'], axis=1).sum(axis=1)
    #     index_order = np.argsort(all_pnl_list[hedging_strategy])
    #     # pnl_order = math.floor(all_pnl_list[hedging_strategy].shape[0]*(100-VaR)/100)
    #     path_num = index_order.iloc[pnl_order]
    #     print(path_num)

    #     hedging_price = perfs[price_from][(perfs[price_from].path_num==path_num)&(perfs[price_from].type.str.contains("hedging_price"))].drop(['path_num','type'], axis=1)
    #     derivative_price = perfs[price_from][(perfs[price_from].path_num==path_num)&(perfs[price_from].type.str.contains("derivative_price"))].drop(['path_num','type'], axis=1)

    #     action_list = {}
    #     holding_list = {}
    #     pnl_list = {}
    #     for n, perf in perfs.items():
    #         action_list[n + " action"] = perf[(perf.path_num==path_num)&(perf.type.str.contains("action"))].drop(['path_num','type'], axis=1)
    #         holding_list[n + " holding"] = perf[(perf.path_num==path_num)&(perf.type.str.contains("holding"))].drop(['path_num','type'], axis=1)
    #         pnl_list[n + " acc. pnl"] = perf[(perf.path_num==path_num)&(perf.type=='pnl')].drop(['path_num','type'], axis=1)

    #     hedging_price.columns = hedging_price.columns.astype(int)
    #     hedging_price = hedging_price.reindex(sorted(hedging_price.columns), axis=1).values
    #     derivative_price.columns = derivative_price.columns.astype(int)
    #     derivative_price = derivative_price.reindex(sorted(derivative_price.columns), axis=1).values
    #     for k in pnl_list.keys():
    #         pnl_list[k].columns = pnl_list[k].columns.astype(int)
    #         pnl_list[k] = pnl_list[k].reindex(sorted(pnl_list[k].columns), axis=1).values[0]
    #     for k in action_list.keys():
    #         action_list[k].columns = action_list[k].columns.astype(int)
    #         action_list[k] = action_list[k].reindex(sorted(action_list[k].columns), axis=1).values
    #     for k in holding_list.keys():
    #         holding_list[k].columns = holding_list[k].columns.astype(int)
    #         holding_list[k] = holding_list[k].reindex(sorted(holding_list[k].columns), axis=1).values
    #     path_plot(pnl_list, action_list, holding_list,
    #             hedging_inst, hedging_price, liability_inst, derivative_price,
    #             f"{hedging_strategy} Path{pnl_order}")
    #     hedging_dim = hedging_price.shape[0]
    #     derivative_dim = derivative_price.shape[0]
    #     step_dim = hedging_price.shape[1]
    #     pnls = {}
    #     for n, p in pnl.items():
    #         pnls[n] = np.cumsum(pd.to_numeric(p))
    #     # no_cum_pnl = np.cumsum(pd.to_numeric(no_pnl))
    #     # d4pg_cum_holding = np.cumsum(d4pg_action, axis=1) + np.expand_dims(initial_stock_holding, -1)
    #     # delta_cum_holding = np.cumsum(delta_action, axis=1) + np.expand_dims(initial_stock_holding, -1)
    #     # no_cum_holding = np.cumsum(no_action, axis=1) + np.expand_dims(initial_stock_holding, -1)

    #     num_sub_plts = 2+hedging_dim*2
    #     fig, axs = plt.subplots(num_sub_plts, 1, figsize=(num_sub_plts*20,20))
    #     for i in range(hedging_dim):
    #         axs[0].plot(hedging_price[i,:], label=f'{hedging_inst[i]} Price', color='orange')
    #     # axs[0].plot(hedging_price[0,:], label='Stock Price', color='orange')
    #     axs[0].grid(True)
    #     axs[0].set_ylabel('Hedging Price')
    #     ax2 = axs[0].twinx()  # instantiate a second axes that shares the same x-axis
    #     ax2.set_ylabel('Liability Prices')
    #     for i in range(derivative_dim):
    #         ax2.plot(derivative_price[i,:], label=f'{liability_inst[i]}', color='blue')
    #     # ax2.plot(derivative_price[0,:], label='European Option Price', color='blue')
    #     axs[0].legend(loc='upper left')
    #     ax2.legend(loc='lower left')

    #     for i in range(hedging_dim):
    #         for n, a in action.items():
    #         try:
    #             axs[1+i*2].bar(range(0, step_dim), a[i,:], alpha=0.5, label=n)
    #         except:
    #             continue
    #         # axs[1+i*2].bar(range(0, step_dim), no_action[i,:], alpha=0.5, label='No hedge bot hedging action')
    #         axs[1+i*2].legend(loc='upper left')
    #         axs[1+i*2].set_ylabel(f'Hedging {i} Action')

    #         for n, h in cum_holding.items():
    #         try:
    #             axs[2*(1+i)].bar(range(0, step_dim), h[i,:], alpha=0.5, label=n)
    #         except:
    #             continue
    #         # axs[2+i*2].bar(range(0, step_dim), no_cum_holding[i,:], alpha=0.5, label='No hedge bot hedging holding')
    #         axs[2*(1+i)].legend(loc='upper left')
    #         axs[2*(1+i)].set_ylabel(f'{hedging_inst[i]} Holding')

    #     for n, l in pnls.items():
    #         axs[1+2*hedging_dim].plot(l, label=n)
    #     # axs[1+2*hedging_dim].plot(no_cum_pnl, label='No hedge bot accumulative P&L')
    #     axs[1+2*hedging_dim].grid(True)
    #     axs[1+2*hedging_dim].legend(loc='upper left')
    #     axs[1+2*hedging_dim].set_xlabel('time')

    #     fig.tight_layout()
    #     plt.savefig(f'{model_path}price_action_path_{figure_name}.png')
    #     plt.show()
