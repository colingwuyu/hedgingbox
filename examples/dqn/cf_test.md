# Results

## Experiment Setup

Hedging 10 shares of short position in a three-month at-the-money call option.

Environment specifictions:

* The stock price evolves by GBM: drift = 10%, sigma=30%. 

* The risk free interest rate is 2%

* Initial stock holding is 5 shares

## Table Exprient Results
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
            <td rowspan=3>BS Option Price SQPenalty P&L</td>
            <td>Mean Total P&L</td>
            <td>1.60</td>
            <td>1.54</td>
            <td rowspan=3><p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_10kappa_moneyness/action.png" styl="max-width:100%"></td>
            <td rowspan=3><p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_10kappa_moneyness/dist.png" styl="max-width:100%"></td>
        </tr>
        <tr>
            <td>Mean Total Reward</td>
            <td>-11.08</td>
            <td>-13.55</td>
        </tr>
        <tr>
            <td>S.D. Total P&L</td>
            <td>5.01</td>
            <td>5.53</td>
        </tr>
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

## Table 2 Exprient Results with Transaction Cost
<table>
    <thead>
        <tr>
            <th>Reward Formula</th>
            <th>Agent</th>
            <th>Transaction Cost</th>
            <th>Delta Hedging Bot</th>
            <th>DQN Hedging Bot</th>
            <th>Action Distribution</th>
            <th>P&L Distribution</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>BS Option Price SQPenalty P&L</td>
            <td rowspan=3>0.1%</td>
            <td>Mean Total P&L</td>
            <td></td>
            <td>0.53</td>
            <td rowspan=3><p style="text-align: center"><image src="01tc_5h_10drift_2r_30sig_10kappa_moneyness/action.png" styl="max-width:100%"></td>
            <td rowspan=3><p style="text-align: center"><image src="01tc_5h_10drift_2r_30sig_10kappa_moneyness/dist.png" styl="max-width:100%"></td>
        </tr>
        <tr>
            <td>Mean Total Reward</td>
            <td></td>
            <td>-14.88</td>
        </tr>
        <tr>
            <td>S.D. Total P&L</td>
            <td></td>
            <td>5.45</td>
        </tr>
        <tr>
            <td rowspan=3>BS Option Price SQPenalty P&L</td>
            <td rowspan=3>0.3%</td>
            <td>Mean Total P&L</td>
            <td></td>
            <td>-0.66</td>
            <td rowspan=3><p style="text-align: center"><image src="03tc/action.png" styl="max-width:100%"></td>
            <td rowspan=3><p style="text-align: center"><image src="03tc/dist.png" styl="max-width:100%"></td>
        </tr>
        <tr>
            <td>Mean Total Reward</td>
            <td></td>
            <td>-18.56</td>
        </tr>
        <tr>
            <td>S.D. Total P&L</td>
            <td></td>
            <td>6.15</td>
        </tr>
        <tr>
            <td rowspan=3>BS Option Price SQPenalty P&L</td>
            <td rowspan=3>0.5%</td>
            <td>Mean Total P&L</td>
            <td></td>
            <td>-2.10</td>
            <td rowspan=3><p style="text-align: center"><image src="05tc/action.png" styl="max-width:100%"></td>
            <td rowspan=3><p style="text-align: center"><image src="05tc/dist.png" styl="max-width:100%"></td>
        </tr>
        <tr>
            <td>Mean Total Reward</td>
            <td></td>
            <td>-20.49</td>
        </tr>
        <tr>
            <td>S.D. Total P&L</td>
            <td></td>
            <td>6.40</td>
        </tr>
        <tr>
            <td rowspan=3>Option Intrinsic Value SQPenalty P&L</td>
            <td rowspan=3>0.5%</td>
            <td>Mean Total P&L</td>
            <td></td>
            <td>-11.77</td>
            <td rowspan=3><p style="text-align: center"><image src="cf_test_images/tc_intrinc_sq_action.png" styl="max-width:100%"></td>
            <td rowspan=3><p style="text-align: center"><image src="cf_test_images/tc_intrinc_sq_dist.png" styl="max-width:100%"></td>
        </tr>
        <tr>
            <td>Mean Total Reward</td>
            <td></td>
            <td>-605.76</td>
        </tr>
        <tr>
            <td>S.D. Total P&L</td>
            <td></td>
            <td>26.04</td>
        </tr>
        <tr>
            <td rowspan=3>Cash Flow SQPenalty</td>
            <td rowspan=3>0.5%</td>
            <td>Mean Total P&L</td>
            <td></td>
            <td></td>
            <td rowspan=3><p style="text-align: center"><image src="cf_test_images/tc_cf_sq_action.png" styl="max-width:100%"></td>
            <td rowspan=3><p style="text-align: center"><image src="cf_test_images/tc_cf_sq_dist.png" styl="max-width:100%"></td>
        </tr>
        <tr>
            <td>Mean Total Reward</td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>S.D. Total P&L</td>
            <td></td>
            <td></td>
        </tr>
    </tbody>
</table>

## Hedging Behaviors

### BS Option Price SQPenalty

<p style="text-align: center;"><mark>BS Option Price P&L with Quadratic Penalty Reward Behavior Sample</mark></p>
<p style="text-align: center"><image src="cf_test_images/bs_sq_sample.png" styl="max-width:100%"></p>

### Option Intrinsic Value SQPenalty

<p style="text-align: center;"><mark>Option Intrinsic Value P&L with Quadratic Penalty Reward Behavior Sample</mark></p>
<p style="text-align: center"><image src="cf_test_images/intric_sq_sample.png" styl="max-width:100%"></p>

## My points

* Under arbitrage free assumption, the sum of Cashflows of a perfect hedging strategy will be zero (assume 0 risk free rate, and Cashflow includes option premium). It is the same for P&L, which is zero under perfect hedging. Actually <img src="https://render.githubusercontent.com/render/math?math=Total Cost = \sum(R_t)=0"> no matter <img src="https://render.githubusercontent.com/render/math?math=R_t"> is cashflow or P&L. It is independent on option pricing model used to calculate intermediate P&L. Because <img src="https://render.githubusercontent.com/render/math?math=V_t"> is cancelled out for <img src="https://render.githubusercontent.com/render/math?math=0<t<T"> when aggregating them together. The experimental proof is that mean total rewards under either cash flow reward or intrinsic value P&L reward equates to mean total P&L under black-scholes option price P&L in [table 1](#Table-Exprient-Results). So to find out the optimal hedging strategy, is to find a stragtegy that makes <img src="https://render.githubusercontent.com/render/math?math=Total Cost = 0">.

* Transfer the optimal hedging problem to MDP. John Hull uses an objective function, which maximizes total cost's mean and minimize total cost's standard deviation.

### John Hull's Objective Function

<p style="text-align: center"><image src="cf_test_images/JH_obj_func.png" styl="max-width:100%">
    
### Our Objective Function

<p style="text-align: center"><image src="cf_test_images/obj_func.png" styl="max-width:100%">

Except maximize expected total cost, the difference comes in the second term. John Hull's second term minimizes <img src="https://render.githubusercontent.com/render/math?math=Total Cost (C_t)">'s standard deviation, but our second term minimizes each step's P&L. John Hull's objective function satisfies general perfect hedging optimization problem, but our objective function applies to a narrower optimal hedging problem, which is the P&L representation with an accurate intermediate option price <img src="https://render.githubusercontent.com/render/math?math=V_t">. Because when <img src="https://render.githubusercontent.com/render/math?math=V_t"> is accurate, a perfect hedging strategy will have zero P&L over all timesteps (as our reward incentive). An obvious proof to show the failure of our objective function is the experiment with reward (Option Intrinsic Value SQPenalty P&L) in [table 1](#Table-Exprient-Results). The experiment results show that RL agent's mean reward outperforms delta hedging agent's mean reward. However, the total cost (total P&l) of RL agent has a much wilder distribution than delta hedging agent.

Some interesting results are shown in [RL hedging behavior](#hedging-behaviors). We see [RL trained with intrinsic value](#Option-Intrinsic-Value-SQPenalty) holds 10 shares when option is in the money (ITM) and holds 0 shares when option is out of the money (OTM). My understanding is our objective function gives incentive to agent for learning the delta (<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial V}{\partial S}">) at each step. Because the P&L reward defines <img src="https://render.githubusercontent.com/render/math?math=N_o*dV %2B N_s*dS"> (if ignoring transaction cost). Our reward encourages agent to make <img src="https://render.githubusercontent.com/render/math?math=PnL=0"> at each step, equvalentlly encourages agent to learn delta for stock holding. Intrinsic value's delta is 0 for OTM and 1 for ITM. Similarly, we see [RL trained with BS optioon price](#BS-Option-Price-SQPenalty) holds almost BS-delta shares.
Here is the P&L reward formula:
<p style="text-align: center"><image src="cf_test_images/pnl_reward.png" styl="max-width:100%">

But John Hull's objective function will suffer from the convergence speed of expected value of quadratic total cost.

* For cash flow reward, there truely exists Credit Assignment Problem (CAP). An obvious proof is that use total cash flow as objective function. It is difficult for RL algo to learn the optimal strategy. 
<p style="text-align: center"><image src="cf_test_images/cash_flow_reward.png" styl="max-width:100%">

The short term incentive is to short stock to get immediate positive reward. But RL cannot infer that shorting stock will accumulate liability, and the shorting positions have to be covered at the terminal step. That will cause a large negative reward ultimately (credit assignment at last). Therefore RL's algo learns to short stocks all the time (see action distribution of DQN agent trained with Cash Flow reward in [table 1](#Table-Exprient-Results)). The optimal strategy should long stock to exploit its 10% drift as RL trained under option intrinsic value P&L reward. This proves it is hard for RL algo to estimate <img src="https://render.githubusercontent.com/render/math?math=E(C_t)"> accurately (not even the second moment of total cost) with cash flow reward.

## My Summary

With the [objective function](#our-objective-function), RL algo tries to learn the delta of option price w.r.t. stock price. This implies the necessary of option pricing model in the reward.
