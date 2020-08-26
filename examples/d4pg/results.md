
# D4PG Bot Results

## Experiment Setups

Hedging 10 shares of short position in a three-month at-the-money call option.

Environment specifictions:

* The stock price evolves by GBM: drift = 10%, sigma=30%. 

* The risk free interest rate is 2%

* Initial stock holding is 5 shares

## Experiment Results

### table 1: Tested Results

<table>
    <thead>
        <tr>
            <th>Transaction Cost</th>
            <th>Statistic</th>
            <th>Delta Hedging Bot*</th>
            <th>D4PG Hedging Bot</th>
            <th>DQN Hedging Bot</th>
            <th>Action Distribution</th>
            <th>P&L Distribution</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>0%</td>
            <td>Mean Total P&L</td>
            <td>1.49</td>
            <td>1.65</td>
            <td>1.54</td>
            <td rowspan=3><p style="text-align: center"><image src="0tc/action.png" styl="max-width:100%"></td>
            <td rowspan=3><p style="text-align: center"><image src="0tc/dist.png" styl="max-width:100%"></td>
        </tr>
        <tr>
            <td>Mean Total Reward</td>
            <td>-9.11</td>
            <td>-9.46</td>
            <td>-13.55</td>
        </tr>
        <tr>
            <td>S.D. Total P&L</td>
            <td>4.51</td>
            <td>4.77</td>
            <td>5.53</td>
        </tr>
        <tr>
            <td rowspan=3>0.1%</td>
            <td>Mean Total P&L</td>
            <td>-0.32</td>
            <td>-0.18</td>
            <td>0.53</td>
            <td rowspan=3><p style="text-align: center"><image src="01tc/action.png" styl="max-width:100%"></td>
            <td rowspan=3><p style="text-align: center"><image src="01tc/dist.png" styl="max-width:100%"></td>
        </tr>
        <tr>
            <td>Mean Total Reward</td>
            <td>-10.97</td>
            <td>-11.29</td>
            <td>-14.88</td>
        </tr>
        <tr>
            <td>S.D. Total P&L</td>
            <td>4.46</td>
            <td>4.54</td>
            <td>5.45</td>
        </tr>
        <tr>
            <td rowspan=3>0.5%</td>
            <td>Mean Total P&L</td>
            <td>-7.58</td>
            <td>-5.94</td>
            <td>-2.10</td>
            <td rowspan=3><p style="text-align: center"><image src="05tc/action.png" styl="max-width:100%"></td>
            <td rowspan=3><p style="text-align: center"><image src="05tc/dist.png" styl="max-width:100%"></td>
        </tr>
        <tr>
            <td>Mean Total Reward</td>
            <td>-20.77</td>
            <td>-20.05</td>
            <td>-20.49</td>
        </tr>
        <tr>
            <td>S.D. Total P&L</td>
            <td>5.28</td>
            <td>4.94</td>
            <td>6.40</td>
        </tr>
    </tbody>
</table>

**Delta Hedging Bot uses continous buy/sell action*

## Training Progress

### 0% Transaction Cost
<p style="text-align: center;"><mark>Traing Progress (Reward)</mark></p>
<p style="text-align: center"><image src="0tc/reward.png"></p>

<p style="text-align: center;"><mark>Traing Progress (Quantiles)</mark></p>
<p style="text-align: center"><image src="0tc/quantiles.png"></p>

<p style="text-align: center;"><mark>Terminal P&L Disctribution</mark></p>
<p style="text-align: center"><image src="0tc/firstaction.png" styl="max-width:100%"></p>

### 0.1% Transaction Cost
<p style="text-align: center;"><mark>Traing Progress (Reward)</mark></p>
<p style="text-align: center"><image src="01tc/reward.png"></p>

<p style="text-align: center;"><mark>Traing Progress (Quantiles)</mark></p>
<p style="text-align: center"><image src="01tc/quantiles.png"></p>

<p style="text-align: center;"><mark>Terminal P&L Disctribution</mark></p>
<p style="text-align: center"><image src="01tc/firstaction.png" styl="max-width:100%"></p>

### 0.5% Transaction Cost
<p style="text-align: center;"><mark>Traing Progress (Reward)</mark></p>
<p style="text-align: center"><image src="05tc/reward.png"></p>

<p style="text-align: center;"><mark>Traing Progress (Quantiles)</mark></p>
<p style="text-align: center"><image src="05tc/quantiles.png"></p>

<p style="text-align: center;"><mark>Terminal P&L Disctribution</mark></p>
<p style="text-align: center"><image src="05tc/firstaction.png" styl="max-width:100%"></p>

## P&L Tail Behaviors

### 0% Transaction Cost

<p style="text-align: center;"><mark>1st Worst case</mark></p>
<p style="text-align: center"><image src="0tc/price_action_prediction_sample0.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>2nd Worst case</mark></p>
<p style="text-align: center"><image src="0tc/price_action_prediction_sample1.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>3rd Worst case</mark></p>
<p style="text-align: center"><image src="0tc/price_action_prediction_sample2.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>4th Worst case</mark></p>
<p style="text-align: center"><image src="0tc/price_action_prediction_sample3.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>5th Worst case</mark></p>
<p style="text-align: center"><image src="0tc/price_action_prediction_sample4.png" styl="max-width:100%"></p>

### 0.1% Transaction Cost

<p style="text-align: center;"><mark>1st Worst case</mark></p>
<p style="text-align: center"><image src="01tc/price_action_prediction_sample0.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>2nd Worst case</mark></p>
<p style="text-align: center"><image src="01tc/price_action_prediction_sample1.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>3rd Worst case</mark></p>
<p style="text-align: center"><image src="01tc/price_action_prediction_sample2.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>4th Worst case</mark></p>
<p style="text-align: center"><image src="01tc/price_action_prediction_sample3.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>5th Worst case</mark></p>
<p style="text-align: center"><image src="01tc/price_action_prediction_sample4.png" styl="max-width:100%"></p>

### 0.5% Transaction Cost

<p style="text-align: center;"><mark>1st Worst case</mark></p>
<p style="text-align: center"><image src="05tc/price_action_prediction_sample0.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>2nd Worst case</mark></p>
<p style="text-align: center"><image src="05tc/price_action_prediction_sample1.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>3rd Worst case</mark></p>
<p style="text-align: center"><image src="05tc/price_action_prediction_sample2.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>4th Worst case</mark></p>
<p style="text-align: center"><image src="05tc/price_action_prediction_sample3.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>5th Worst case</mark></p>
<p style="text-align: center"><image src="05tc/price_action_prediction_sample4.png" styl="max-width:100%"></p>

