# Results

The environment reward setup:

<p style="text-align: center"><image src="reward_formula.png"></p>

---
## Test Scenarios

<!--ts-->
   * [Scenario 1 - No Trading Cost with neutral delta at initial state](#scenario-1)
   * [Scenario 2 - Trading Cost with neutral delta at initial state](#scenario-2)
   * [Scenario 3 - No Trading Cost with negative delta at initial state](#scenario-3)
   * [Scenario 4 - Trading Cost with negative delta at initial state](#scenario-4)
   * [Scenario 5 - Large trading Cost with negative delta at initial state](#scenario-5)
<!--te-->

## Scenario 1

    * Trading cost = 0
    * Call option holding = -10 shares 
    * Initial stock holding = 5 shares

<p style="text-align: center;"><mark>Traing Progress (Reward)</mark></p>
<p style="text-align: center"><image src="zerotradingcost_kappa8_init5holding/reward.png"></p>

<p style="text-align: center;"><mark>Traing Progress (Quantiles)</mark></p>
<p style="text-align: center"><image src="zerotradingcost_kappa8_init5holding/quantiles.png"></p>

<p style="text-align: center;"><mark>Action Disctribution</mark></p>
<p style="text-align: center"><image src="zerotradingcost_kappa8_init5holding/action.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>Terminal P&L Disctribution</mark></p>
<p style="text-align: center"><image src="zerotradingcost_kappa8_init5holding/dist.png" styl="max-width:100%"></p>

[Go Back](#test-scenarios)

## Scenario 2

    * Trading cost = 1% 
    * Call option holding = -10 shares
    * Initial stock holding = 5 shares

<p style="text-align: center;"><mark>Traing Progress (Reward)</mark></p>
<p style="text-align: center"><image src="1pcttradingcost_kappa8_init5holding/reward.png"></p>

<p style="text-align: center;"><mark>Traing Progress (Quantiles)</mark></p>
<p style="text-align: center"><image src="1pcttradingcost_kappa8_init5holding/quantiles.png"></p>

<p style="text-align: center;"><mark>Action Disctribution</mark></p>
<p style="text-align: center"><image src="1pcttradingcost_kappa8_init5holding/action.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>Terminal P&L Disctribution</mark></p>
<p style="text-align: center"><image src="1pcttradingcost_kappa8_init5holding/dist.png" styl="max-width:100%"></p>

[Go Back](#test-scenarios)

## Scenario 3

    * Trading cost = 0%
    * Call option holding = -10 shares
    * Initial stock holding = 2 shares

<p style="text-align: center;"><mark>Traing Progress (Reward)</mark></p>
<p style="text-align: center"><image src="zerotradingcost_kappa8_init2holding/reward.png"></p>

<p style="text-align: center;"><mark>Traing Progress (Quantiles)</mark></p>
<p style="text-align: center"><image src="zerotradingcost_kappa8_init2holding/quantiles.png"></p>

<p style="text-align: center;"><mark>Initial State Action Q-Values</mark></p>
<p style="text-align: center"><image src="zerotradingcost_kappa8_init2holding/firstaction.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>Action Disctribution</mark></p>
<p style="text-align: center"><image src="zerotradingcost_kappa8_init2holding/action.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>Terminal P&L Disctribution</mark></p>
<p style="text-align: center"><image src="zerotradingcost_kappa8_init2holding/dist.png" styl="max-width:100%"></p>

[Go Back](#test-scenarios)

## Scenario 4

    * Trading cost = 1%
    * Call option holding = -10 shares
    * Initial stock holding = 2 shares

<p style="text-align: center;"><mark>Traing Progress (Reward)</mark></p>
<p style="text-align: center"><image src="1pcttradingcost_kappa8_init2holding/reward.png"></p>

<p style="text-align: center;"><mark>Traing Progress (Quantiles)</mark></p>
<p style="text-align: center"><image src="1pcttradingcost_kappa8_init2holding/quantiles.png"></p>

<p style="text-align: center;"><mark>Initial State Action Q-Values</mark></p>
<p style="text-align: center"><image src="1pcttradingcost_kappa8_init2holding/firstaction.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>Action Disctribution</mark></p>
<p style="text-align: center"><image src="1pcttradingcost_kappa8_init2holding/action.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>Terminal P&L Disctribution</mark></p>
<p style="text-align: center"><image src="1pcttradingcost_kappa8_init2holding/dist.png" styl="max-width:100%"></p>

[Go Back](#test-scenarios)

## Scenario 5

    * Trading cost = 5%
    * Call option holding = -10 shares
    * Initial stock holding = 2 shares

<p style="text-align: center;"><mark>Traing Progress (Reward)</mark></p>
<p style="text-align: center"><image src="5pcttradingcost_kappa8_init2holding/reward.png"></p>

<p style="text-align: center;"><mark>Traing Progress (Quantiles)</mark></p>
<p style="text-align: center"><image src="5pcttradingcost_kappa8_init2holding/quantiles.png"></p>

<p style="text-align: center;"><mark>Initial State Action Q-Values</mark></p>
<p style="text-align: center"><image src="5pcttradingcost_kappa8_init2holding/firstaction.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>Action Disctribution</mark></p>
<p style="text-align: center"><image src="5pcttradingcost_kappa8_init2holding/action.png" styl="max-width:100%"></p>

<p style="text-align: center;"><mark>Terminal P&L Disctribution</mark></p>
<p style="text-align: center"><image src="5pcttradingcost_kappa8_init2holding/dist.png" styl="max-width:100%"></p>

[Go Back](#test-scenarios)
