# Results

## Experiment Setup

Hedging 10 shares of short position in a three-month at-the-money call option.

Environment specifictions:

* The stock price evolves by GBM: drift = 10%, sigma=30%. 

* The risk free interest rate is 2%

* Initial stock holding is 5 shares

## Table 1 distributions of Hedging P&L and Reward
<table>
    <thead>
        <tr>
            <th>Reward Formula</th>
            <th>Agent</th>
            <th>Delta Hedging Bot</th>
            <th>DQN Hedging Bot</th>
            <th>Action Distribution</th>
            <th>P&L Distribution</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>Option Intrinsic Value P&L</td>
            <td>Mean Total P&L</td>
            <td>1.60</td>
            <td>279.61</td>
            <td rowspan=3><p style="text-align: center"><image src="cf_test_images/intric_action.png" styl="max-width:100%"></td>
            <td rowspan=3><p style="text-align: center"><image src="cf_test_images/intric_dist.png" styl="max-width:100%"></td>
        </tr>
        <tr>
            <td>Mean Total Reward</td>
            <td>1.60</td>
            <td>279.60</td>
        </tr>
        <tr>
            <td>S.D. Total P&L</td>
            <td>5.01</td>
            <td>1,456.72</td>
        </tr>
        <tr>
            <td rowspan=3>Option Intrinsic Value SQPenalty P&L</td>
            <td>Mean Total P&L</td>
            <td>1.60</td>
            <td>0.38</td>
            <td rowspan=3><p style="text-align: center"><image src="cf_test_images/intric_sq_action.png" styl="max-width:100%"></td>
            <td rowspan=3><p style="text-align: center"><image src="cf_test_images/intric_sq_dist.png" styl="max-width:100%"></td>
        </tr>
        <tr>
            <td>Mean Total Reward</td>
            <td>-625.24</td>
            <td>-605.53</td>
        </tr>
        <tr>
            <td>S.D. Total P&L</td>
            <td>5.01</td>
            <td>22.74</td>
        </tr>
        <tr>
            <td rowspan=3>Cash Flow</td>
            <td>Mean Total P&L</td>
            <td>1.60</td>
            <td>-278.29</td>
            <td rowspan=3><p style="text-align: center"><image src="cf_test_images/cf_action.png" styl="max-width:100%"></td>
            <td rowspan=3><p style="text-align: center"><image src="cf_test_images/cf_dist.png" styl="max-width:100%"></td>
        </tr>
        <tr>
            <td>Mean Total Reward</td>
            <td>1.60</td>
            <td>-278.29</td>
        </tr>
        <tr>
            <td>S.D. Total P&L</td>
            <td>5.01</td>
            <td>1,562.08</td>
        </tr>
        <tr>
            <td rowspan=3>Cash Flow SQPenalty</td>
            <td>Mean Total P&L</td>
            <td>1.60</td>
            <td>0.19</td>
            <td rowspan=3><p style="text-align: center"><image src="cf_test_images/cf_sq_action.png" styl="max-width:100%"></td>
            <td rowspan=3><p style="text-align: center"><image src="cf_test_images/cf_sq_dist.png" styl="max-width:100%"></td>
        </tr>
        <tr>
            <td>Mean Total Reward</td>
            <td>-132,189.76</td>
            <td>-35,850.98</td>
        </tr>
        <tr>
            <td>S.D. Total P&L</td>
            <td>5.01</td>
            <td>23.69</td>
        </tr>
    </tbody>
</table>

## Hedging Behaviors
<p style="text-align: center;"><mark>BS Option Price P&L with Quadratic Penalty Reward Behavior Sample</mark></p>
<p style="text-align: center"><image src="cf_test_images/bs_sq_sample.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>Option Intrinsic Value P&L with Quadratic Penalty Reward Behavior Sample</mark></p>
<p style="text-align: center"><image src="cf_test_images/intric_sq_sample.png" styl="max-width:100%"></p>

## My points

* Under arbitrage free assumption, the sum of Cashflowsof a perfect hedging strategy will be zero (assume 0 risk free rate, and CF includes option premium). It is the same for P&L, which is zero under perfect hedging. <img src="https://render.githubusercontent.com/render/math?math=Total Cost = \sum(R_t)=0"> no matter <img src="https://render.githubusercontent.com/render/math?math=R_t"> is cashflow or P&L. And it is independent with option pricing model used to calculate intermediate P&L. Because <img src="https://render.githubusercontent.com/render/math?math=V_t"> is cancelled out for <img src="https://render.githubusercontent.com/render/math?math=0<t<T">. So to find out the optimal hedging strategy, is to find stragtegy so that <img src="https://render.githubusercontent.com/render/math?math=Total Cost = 0">.

* Transfer the perfect hedging optimization problem to MDP. John Hull uses the objective function, which maximizes total cost's mean and minimize total cost's sandard deviation.
<p style="text-align: center"><image src="cf_test_images/JH_obj_func.png" styl="max-width:100%">

Our objective function
<p style="text-align: center"><image src="cf_test_images/obj_func.png" styl="max-width:100%">

The difference is that John Hull's objective function minimizes <img src="https://render.githubusercontent.com/render/math?math=Total Cost">'s standard deviation, but our objective function minimizes P&L at each step. John Hull's objective function satisfies general perfect hedging optimization problem, but our objective function only applies to P&L depending on the accuracy of intermediate option price <img src="https://render.githubusercontent.com/render/math?math=V_t">. When <img src="https://render.githubusercontent.com/render/math?math=V_t"> is accurate, a perfect hedging strategy will imply to zero P&L over all timesteps. An obvious proof that our objective function does not work is the experiment with P&L calculated by intrinsic value. RL agent's mean reward outperforms delta hedging agent's mean reward. However, the total cost (total P&l) of RL agent has a wild distribution than delta hedging agent.
We see the behavior of RL with intrinsic value reward holds 10 shares when option is in the money (ITM) and holds 0 shares when option is out of the money (OTM). My understanding is our objective function gives incentive to agent for learning the delta (<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial V}{\partial S}">) at each step. Because the P&L reward defines <img src="https://render.githubusercontent.com/render/math?math=N_o*dV + N_s*dS"> (assuming no transaction cost). Our reward encourages agent to make <img src="https://render.githubusercontent.com/render/math?math=PnL=0"> at each step, equvalentlly encourages agent to learn delta for stock holding. Intrinsic value's delta is 0 for OTM and 1 for ITM.

<p style="text-align: center"><image src="cf_test_images/pnl_reward.png" styl="max-width:100%">

But John Hull's objective function will suffer from the convergence speed of expected value of quadratic total cost.

* For cash flow reward, there truely exists Credit Assignment Problem (CAP). An obvious proof is that use total cash flow as objective function. It is difficult for RL algo to learn. 
<p style="text-align: center"><image src="cf_test_images/cash_flow_reward.png" styl="max-width:100%">

The short term incentive is to short stock to get immediate positive reward. But RL does not understand shorting stock will accumulate liability and a large negative reward at terminal step (credit assignment at last). This makes RL's behavior to short stocks all the time. Under cash flow reward, I expect RL learns to long stock in order to exploit its 10% drift as under option intrinsic value P&L reward. This proves RL algo cannot estimate <img src="https://render.githubusercontent.com/render/math?math=E(C_t)"> accurately (not even the second moment of total cost).
