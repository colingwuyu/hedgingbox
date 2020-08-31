Quadratic Penalty Reward
$$R_t=PnL_t-\kappa/2(PnL^2)$$

Mean-Variance Risk Measure Objective Function
$$F = E[\sum{PnL}]-c\sqrt{Var[\sum{PnL}]}$$

Here $PnL$ contains model dependent OTC derivative price at each time $t$. To test the impact of OTC derivative pricing model accuracy, we apply shocks onto each step's OTC derivative price to represent the model errors. 

Twp types of shocks:

- a random shock: $e\sim N(\mu, \sigma)$. $\mu$ and $\sigma$ are a pecentage of the derivative's model price $S^{opt}_t$
- systematic shock: static percentage shock of the derivative's model price

In the tests, 0.5% transaction cost is used. For a fair comparison, D4PG Bot is trained in each environment with 20k episodes. The following is the summary of terminal P&L distribution of the prediction set (1k episodes).

<table>
    <thead>
        <tr>
            <th>Test Enviroenment</th>
            <th>Objective Function</th>
            <th>Shock Definition</th>
            <th>P&L Mean</th>
            <th>P&L S.D.</th>
            <th>P&L 95% VaR</th>
            <th>Mean-Variance Risk Measure</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>E1</td>
            <td>Quadratic Penalty Reward</td>
            <td>No Shock</td>
            <td>-5.93</td>
            <td>5.25</td>
            <td>-15.79</td>
            <td>-13.81</td>
        </tr>
        <tr>
            <td>E2</td>
            <td>Mean-Variance Risk Measure</td>
            <td>No Shock</td>
            <td>-3.93</td>
            <td>5.25</td>
            <td>-12.73</td>
            <td>-11.80</td>
        </tr>
        <tr>
            <td>E3</td>
            <td>Mean-Variance Risk Measure</td>
            <td>Random shock miu=0, sigma=5%S<sub>t</sub><sup>opt</sup></td>
            <td>-5.06</td>
            <td>5.45</td>
            <td>-11.71</td>
            <td>-13.23</td>
        </tr>
        <tr>
            <td>E4</td>
            <td>Mean-Variance Risk Measure</td>
            <td>Random shock miu=5%S<sub>t</sub><sup>opt</sup>, sigma=3%S<sub>t</sub><sup>opt</sup></td>
            <td>-4.85</td>
            <td>4.98</td>
            <td>-13.76</td>
            <td>-12.31</td>
        </tr>
        <tr>
            <td>E5</td>
            <td>Mean-Variance Risk Measure</td>
            <td>Random shock miu=-5%S<sub>t</sub><sup>opt</sup>, sigma=3%S<sub>t</sub><sup>opt</sup></td>
            <td>-4.79</td>
            <td>5.37</td>
            <td>-14.23</td>
            <td>-12.84</td>
        </tr>
        <tr>
            <td>E6</td>
            <td>Mean-Variance Risk Measure</td>
            <td>Systematic shock=5%S<sub>t</sub><sup>opt</sup></td>
            <td>-3.77</td>
            <td>5.48</td>
            <td>-13.21</td>
            <td>-11.99</td>
        </tr>
        <tr>
            <td>E7</td>
            <td>Mean-Variance Risk Measure</td>
            <td>Systematic shock=-5%S<sub>t</sub><sup>opt</sup></td>
            <td>-4.05</td>
            <td>5.18</td>
            <td>-12.48</td>
            <td>-11.81</td>
        </tr>
        <tr>
            <td>E8</td>
            <td>Mean-Variance Risk Measure</td>
            <td>Systematic shock=0.5%S<sub>t</sub><sup>opt</sup></td>
            <td>-3.83</td>
            <td>5.24</td>
            <td>-12.35</td>
            <td>-11.68</td>
        </tr>
    </tbody>
</table>

# My idea

An accurate option price is the PV of perfect/optimal hedging costs in future. Using an option price in reward is equivalent to distribute the total return back to each step. It gives more help and incentives for agent's objective function bootstrap, if the option price is more accurate.

It is similar to Inverse RL / imitation learning / apprenticeship learning. IRL learns the rewards/cost function from expert demonstrations (a close to optimal policy), when the rewards are not obviously structured for each action. An example of IRL is autonomous driving, that learns action rewards from experienced taxi driver. The intermediate reward to autonomous driving is not treavially formulated, but it only has a treavial final reward (such as arrive destination safely and quickly without any accident). IRL learns the intermediate action reward function from expert/optimal policy (taxi driver). After IRL defines rewards, RL agent imitating the expert behavior and sometimes after training, it exceeds the performance of the expert. I think is is also one way to solve CAP. When credit assignment of an acction under a state is not clear, agent can imitate expert's demonstration to learn behavior's credit assignment.

It gives me inspiration that we have some existing pricing model for exotic OTC derivatives. These pricing models give optimal hedging policy under some underlying dynamic assumptions and frictionless market. Use such price to construct reward for some other environment (such as with transaction cost, and other underlying dynamics). It solves CAP and inspires RL agent to learn towards the objective.

From the test results, we can observe that RL agent has best performance with a systematic shock of 0.5%. Here 0.5% is the transaction cost percentage. Because BS option price is under assumption of 0 transaction cost, which is underestimated. Therefore, the price is more close to accurate price under the training environment after the 0.5% adjustment. It is very like to trader. the more accurate the model price, model hedging ratios and model based daily P&L with trader's expertise adjustments, the better the hedging performance will be achieved.

The test results also inspire me that RL learning algo has some level of tolerance on the accuracy of pricing model. It still learns quite good results even with 5%~10% pricing errors. With more advanced RL techniques, such as some novel exploration techniques for state space visitation, I believe RL agent can perform better and have higher tolerance level.

It also gives me inspiration that we can use some estimation, such as DNN pricing models, in environment simulator to accelerate the exotic derivatives pricing.