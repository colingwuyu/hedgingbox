
# DQN Bot Results

## Experiment Setups

Hedging 10 shares of short position in a three-month at-the-money call option.

Environment specifictions:

* The stock price evolves by GBM: drift = 10%, sigma=30%. 

* The risk free interest rate is 2%

* Initial stock holding is 5 shares

First to discover performance comparison between the DQN hedging bot (model-free agent) and the model-dependent Delta hedging bot performance **without any transaction cost**. Table 1 presents the trained DQN hedging bots with different Q-Value DNN architecture, reward definition and environment states.  

### table 1: Tested DQN hedging bots' definitions  

| DQN Agent | Q-Value DNN Architecture | Reward Kappa | State Variables |
|:--:|:--:|:--:|:--|
|DQN Hedging Bot 1| ForwardFeed NN|1.0|Stock price, Stock holding, Expiry time|
|DQN Hedging Bot 2| Duelling DNN|1.0|Stock price, Stock holding, Expiry time, Option price|
|DQN Hedging Bot 3| Duelling DNN|0.6|Stock price, Stock holding, Expiry time, Option price, Moneyness*|
|DQN Hedging Bot 4| Dueling DNN|1.0|Stock price, Stock holding, Expiry time, Option price, Moneyness|

* *The training process uses exploration epsilon decay (0.8->0.5->0.2) every 50k episodes (400k learning). Learning rate is 1e-3.*
* **Moneyness=Forward Price/Option Strike*

## Experiment Results

All hedging bots contest with same 1000 simulated paths. For each path, the accumulative daily P&L is calculated. The distributional statistics of these 1000 P&Ls are presented in Table 2.

### table 2: Total P&L Stats Comparison

| Agent | Mean Total P&L | S.D. Total P&L | 5% Quantile | 10% Quantile | Mean Reward (kappa=1.0)* |
|:-----|:--------------:|:--------------:|:-----------:|:-----------:|:-----------:|
|Delta Hedging Bot| 1.597 | 5.015 | -6.964 | -4.625 | -11.079|
|DQN Hedging Bot 1|  ||||
|DQN Hedging Bot 2| 1.370 | 5.693 | -5.703 | -8.146 | -14.862 |
|DQN Hedging Bot 3| 1.685 | 5.478 | -5.155 | -8.181 | -15.384|
|DQN Hedging Bot 4| 1.543 | 5.534 | -5.071 | -7.508 | -13.552 |

* **Mean Reward is the mean of 1000 paths' rewards*
* *DQN Hedging Bot 4 has achieved best P&L std = 5.257, in which case P&L mean = 1.261, 5% Quantile = -7.394, 10% Quantile = -5.138 and Mean Reward = -14.03*

## Trading Cost

Introduce 0.1% transaction cost into the environment.

### table 3: Total P&L comparison

 Agent | Mean Total P&L | S.D. Total P&L | 5% Quantile | 10% Quantile | Mean Reward (kappa=1.0)* |
|:-----|:--------------:|:--------------:|:-----------:|:-----------:|:-----------:|
|Delta Hedging Bot| 0.093 | 5.459 | -8.576 | -6.320 | -12.591 |
|DQN Hedging Bot*| 0.526 | 5.453 | -8.944 | -6.089 | -14.88 |

**DQN Hedging Bot uses duelling heads, trained with reward kappa=1.0 and state variables including stock price, stock holding, expiry time and moneyness*

## Training Progress

### DQN Hedging Bot 2

Duelling DNN; Reward kappa=1.0; State excludes moneyness.

<p style="text-align: center;"><mark>Traing Progress (Reward)</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_10kappa/reward.png"></p>

<p style="text-align: center;"><mark>Traing Progress (Quantiles)</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_10kappa/quantiles.png"></p>

<p style="text-align: center;"><mark>Action Disctribution</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_10kappa/action.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>Terminal P&L Disctribution</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_10kappa/dist.png" styl="max-width:100%"></p>

### DQN Hedging Bot 3

Duelling DNN; Reward kappa=0.6; State includes moneyness.

<p style="text-align: center;"><mark>Traing Progress (Reward)</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_06kappa_moneyness/reward.png"></p>

<p style="text-align: center;"><mark>Traing Progress (Quantiles)</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_06kappa_moneyness/quantiles.png"></p>

<p style="text-align: center;"><mark>Action Disctribution</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_06kappa_moneyness/action.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>Terminal P&L Disctribution</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_06kappa_moneyness/dist.png" styl="max-width:100%"></p>

### DQN Hedging Bot 4

Duelling DNN; Reward kappa=1.0; State includes moneyness. 

<p style="text-align: center;"><mark>Traing Progress (Reward)</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_10kappa_moneyness/reward.png"></p>

<p style="text-align: center;"><mark>Traing Progress (Quantiles)</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_10kappa_moneyness/quantiles.png"></p>

<p style="text-align: center;"><mark>Action Disctribution</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_10kappa_moneyness/action.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>Terminal P&L Disctribution</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_10kappa_moneyness/dist.png" styl="max-width:100%"></p>

### DQN Hedging Bot with Transaction Cost

Duelling DNN; Reward kappa=1.0; State includes moneyness; Transaction cost=0.1%

<p style="text-align: center;"><mark>Traing Progress (Reward)</mark></p>
<p style="text-align: center"><image src="01tc_5h_10drift_2r_30sig_10kappa_moneyness/reward.png"></p>

<p style="text-align: center;"><mark>Traing Progress (Quantiles)</mark></p>
<p style="text-align: center"><image src="01tc_5h_10drift_2r_30sig_10kappa_moneyness/quantiles.png"></p>

<p style="text-align: center;"><mark>Action Disctribution</mark></p>
<p style="text-align: center"><image src="01tc_5h_10drift_2r_30sig_10kappa_moneyness/action.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>Terminal P&L Disctribution</mark></p>
<p style="text-align: center"><image src="01tc_5h_10drift_2r_30sig_10kappa_moneyness/dist.png" styl="max-width:100%"></p>


## P&L Tail Behaviors

### DQN Hedging Bot 2

<p style="text-align: center;"><mark>1st Worst case</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_10kappa/price_action_prediction_sample0.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>2nd Worst case</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_10kappa/price_action_prediction_sample1.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>3rd Worst case</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_10kappa/price_action_prediction_sample2.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>4th Worst case</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_10kappa/price_action_prediction_sample3.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>5th Worst case</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_10kappa/price_action_prediction_sample4.png" styl="max-width:100%"></p>

### DQN Hedging Bot 3

<p style="text-align: center;"><mark>1st Worst case</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_06kappa_moneyness/price_action_prediction_sample0.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>2nd Worst case</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_06kappa_moneyness/price_action_prediction_sample1.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>3rd Worst case</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_06kappa_moneyness/price_action_prediction_sample2.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>4th Worst case</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_06kappa_moneyness/price_action_prediction_sample3.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>5th Worst case</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_06kappa_moneyness/price_action_prediction_sample4.png" styl="max-width:100%"></p>

### DQN Hedging Bot 4

<p style="text-align: center;"><mark>1st Worst case</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_10kappa_moneyness/price_action_prediction_sample0.png" styl="max-width:100%"></p>


<p style="text-align: center;"><mark>2nd Worst case</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_10kappa_moneyness/price_action_prediction_sample1.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>3rd Worst case</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_10kappa_moneyness/price_action_prediction_sample2.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>4th Worst case</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_10kappa_moneyness/price_action_prediction_sample4.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>5th Worst case</mark></p>
<p style="text-align: center"><image src="0tc_5h_10drift_2r_30sig_10kappa_moneyness/price_action_prediction_sample5.png" styl="max-width:100%"></p>

### DQN Hedging Bot with Transaction cost

<p style="text-align: center;"><mark>1st Worst case</mark></p>
<p style="text-align: center"><image src="01tc_5h_10drift_2r_30sig_10kappa_moneyness/price_action_prediction_sample0.png" styl="max-width:100%"></p>


<p style="text-align: center;"><mark>2nd Worst case</mark></p>
<p style="text-align: center"><image src="01tc_5h_10drift_2r_30sig_10kappa_moneyness/price_action_prediction_sample1.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>3rd Worst case</mark></p>
<p style="text-align: center"><image src="01tc_5h_10drift_2r_30sig_10kappa_moneyness/price_action_prediction_sample2.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>4th Worst case</mark></p>
<p style="text-align: center"><image src="01tc_5h_10drift_2r_30sig_10kappa_moneyness/price_action_prediction_sample4.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>5th Worst case</mark></p>
<p style="text-align: center"><image src="01tc_5h_10drift_2r_30sig_10kappa_moneyness/price_action_prediction_sample5.png" styl="max-width:100%"></p>


## Apply trained transaction cost DQN Bot to a 30d call option

Let's use DQN Bot to hedge call options with 30-day maturity.

### table 4 P&L distributional statistics comparison

| Agent | Mean Total P&L | S.D. Total P&L | 5% Quantile | 10% Quantile | Mean Reward (kappa=1.0)* |
|:-----|:--------------:|:--------------:|:-----------:|:-----------:|:-----------:|
|Delta Hedging Bot| -0.169 | 4.612 | -7.470 | -5.709 | -11.135 |
|DQN Hedging Bot| -0.09 | 5.601 | -9.948 | -6.939 | -15.829 |

<p style="text-align: center;"><mark>Terminal P&L Disctribution</mark></p>
<p style="text-align: center"><image src="01tc_5h_10drift_2r_30sig_10kappa_moneyness/30d_dist.png" styl="max-width:100%"></p>
 
If we emphasize the training on hedging 30-day calls, the trained policy will be quickly distorted. The agent cannot hedge 90-day calls efficiently anymore.
