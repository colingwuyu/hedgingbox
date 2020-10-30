import matplotlib.pyplot as plt
import numpy as np
from hb.bots import bot
from typing import List

def _get_colors(num_plots):
    return plt.cm.Accent(np.linspace(0,1,num_plots))

def acc_pnl(bots: List[bot.Bot]):
    colors = _get_colors(len(bots))
    plt.figure()
    for i, bot in enumerate(bots):
        plt.plot(np.cumsum(bot.get_predictor().get_episode_pnl_path()), label=f"{bot.get_name()} Acc. PnL", color=colors[i])
    plt.legend()
    plt.xlabel("Num of Days")
    plt.ylabel("Acc. PnL")
    plt.title("Accumulative PnL")
    return plt
