# Track A GPU Explainability Supplement

base_protocol: `AP-V1-TRACKA-20260923-A1`
diagnostic_protocol: `AP-V1-TRACKA-20260923-A1-DIAG1`
supplement_protocol: `AP-V1-TRACKA-20260923-A1-DIAG1-GPU1`
result_label: `POST_LOCK_DIAGNOSTIC_ONLY`
post_lock: `true`
decision_eligible: `false`
infrastructure_recovery: `true`
a1_decision: `RETAIN_BASELINE`

DIAG1 originally and correctly recorded `NOT_EVALUABLE_INFRASTRUCTURE_FAILURE_CUDA_UNAVAILABLE`. GPU capability recovered later; this GPU1 supplement used only frozen, read-only inference with every XGBoost booster set to `device=cuda:0`. No fitting, training, updating, calibration selection, or scientific retuning occurred.

## GPU qualification and TreeSHAP

- Device: NVIDIA GeForce RTX 4090 Laptop GPU; physical device count: 1.
- Native B4 TreeSHAP additivity maximum errors: POLICY `1.91e-06`, LOCKED `2.86e-06`.
- Contributions explain frozen B4 model margins, not causal effects.

## Global grouped attribution

| feature_group | mean_abs_shap_policy | mean_abs_shap_locked | importance_rank_policy | importance_rank_locked | normalized_attribution_drift |
|---|---|---|---|---|---|
| account_history | 1.2986 | 1.2541 | 1 | 1 | 0.0176 |
| calendar | 0.9450 | 1.0547 | 2 | 2 | 0.0263 |
| product_history | 0.6502 | 0.6395 | 3 | 3 | 0.0056 |
| sector_history | 0.2386 | 0.3254 | 5 | 4 | 0.0229 |
| firmographics | 0.3682 | 0.2763 | 4 | 5 | 0.0268 |
| product | 0.0837 | 0.0880 | 6 | 6 | 0.0008 |

## Top LOCKED raw features

| raw_feature | feature_group | mean_abs_shap_locked | importance_rank_locked | rank_change |
|---|---|---|---|---|
| account_prior_opportunities | account_history | 1.3017 | 1 | 0 |
| engage_quarter | calendar | 1.0725 | 2 | 0 |
| product_prior_opportunities | product_history | 0.3206 | 3 | 0 |
| sector_prior_avg_won_value | sector_history | 0.2909 | 4 | -1 |
| product_prior_resolved | product_history | 0.2621 | 5 | 1 |
| product_prior_avg_won_value | product_history | 0.1452 | 6 | -2 |
| product_prior_win_rate | product_history | 0.1374 | 7 | 1 |
| sector_prior_resolved | sector_history | 0.1351 | 8 | -2 |
| sector_prior_win_rate | sector_history | 0.1152 | 9 | -2 |
| revenue | firmographics | 0.1110 | 10 | -2 |
| days_since_prior_opportunity | account_history | 0.1046 | 11 | 4 |
| company_age | firmographics | 0.0894 | 12 | 3 |

## Local predefined cases

Local explanations cover only predefined missed whales, false priorities, and largest frozen B1/AP-EV rank disagreements. The tidy artifact contains the five largest positive and negative margin contributions per case. Nothing was selected after inspecting SHAP values.

| case_type | opportunity_id | actual_outcome | actual_value | encoded_feature | shap_value |
|---|---|---|---|---|---|
| LARGEST_B1_APEV_DISAGREEMENT | H3K2E35I | WON | 27971.0000 | company_age | -0.1181 |
| LARGEST_B1_APEV_DISAGREEMENT | H3K2E35I | WON | 27971.0000 | product_prior_avg_won_value | -0.2268 |
| LARGEST_B1_APEV_DISAGREEMENT | H3K2E35I | WON | 27971.0000 | account_prior_opportunities | -0.9552 |
| LARGEST_B1_APEV_DISAGREEMENT | H3K2E35I | WON | 27971.0000 | engage_quarter_1 | -0.9869 |
| LARGEST_B1_APEV_DISAGREEMENT | H3K2E35I | WON | 27971.0000 | product_prior_opportunities | -1.7084 |
| LARGEST_B1_APEV_DISAGREEMENT | H3K2E35I | WON | 27971.0000 | product_prior_win_rate | 0.0972 |
| LARGEST_B1_APEV_DISAGREEMENT | H3K2E35I | WON | 27971.0000 | sales_price | 0.1244 |
| LARGEST_B1_APEV_DISAGREEMENT | H3K2E35I | WON | 27971.0000 | sector_prior_resolved | 0.1358 |
| LARGEST_B1_APEV_DISAGREEMENT | H3K2E35I | WON | 27971.0000 | account_prior_avg_won_value | 0.2031 |
| LARGEST_B1_APEV_DISAGREEMENT | H3K2E35I | WON | 27971.0000 | sector_prior_avg_won_value | 0.4035 |
| MISSED_WHALE | H3K2E35I | WON | 27971.0000 | engage_quarter_1 | -0.9869 |
| MISSED_WHALE | H3K2E35I | WON | 27971.0000 | account_prior_opportunities | -0.9552 |

## AP-EV components and discount behavior

The exact two-bin frozen computation was reconstructed with `rho=0.005`; the stored score was not changed. The earlier finding—full ValueCapture@10% 24.23% versus 25.32% without discount—remains explanatory only.

### Absolute-value scaling versus proportional timing

The absolute discount penalty has Spearman rho `0.8682` with sales price. Because the absolute penalty is denominated in value units, that relationship includes opportunity-value scale. The normalized relative penalty has sales-price rho `0.0538` and expected-component-days rho `1.0000`. The near-zero normalized association does not support a claim that high-price opportunities were disproportionately penalized. The perfect monotone relationship with expected component days reflects the fixed positive `rho` applied to the two-bin timing mixture; it identifies timing-associated proportional discounting, not a causal effect.

| penalty_measure | summary_dimension | spearman_correlation |
|---|---|---|
| absolute_discount_penalty | sales_price | 0.8682 |
| absolute_discount_penalty | realized_value | 0.3247 |
| absolute_discount_penalty | expected_component_days | 0.1800 |
| absolute_discount_penalty | undiscounted_total | 0.9850 |
| absolute_discount_penalty | conditional_expected_value | 0.8448 |
| relative_discount_penalty | sales_price | 0.0538 |
| relative_discount_penalty | realized_value | 0.0224 |
| relative_discount_penalty | expected_component_days | 1.0000 |
| relative_discount_penalty | undiscounted_total | 0.0365 |
| relative_discount_penalty | conditional_expected_value | 0.0488 |

The opportunity-level artifact records `absolute_discount_penalty`, `relative_discount_penalty`, `effective_discount_factor`, `expected_component_days`, and the directly available stored `conditional_expected_value` for every LOCKED opportunity. The denominator is `max(undiscounted_total, 1e-12)`. Realized-value quartile cutpoints retain ties, so empty middle quartiles are possible when zero realized values dominate.

| summary_dimension | summary_value | support_n | mean_absolute_discount_penalty | mean_relative_discount_penalty | mean_expected_component_days |
|---|---|---|---|---|---|
| sales_price_quartile | Q1 | 151.0000 | 2.0083 | 0.1191 | 25.8782 |
| sales_price_quartile | Q2 | 60.0000 | 8.5158 | 0.1207 | 26.2525 |
| sales_price_quartile | Q3 | 137.0000 | 27.3687 | 0.1236 | 26.9137 |
| sales_price_quartile | Q4 | 55.0000 | 29.2220 | 0.1180 | 25.6094 |
| realized_value_quartile | Q1 | 215.0000 | 13.4413 | 0.1202 | 26.1378 |
| realized_value_quartile | Q3 | 87.0000 | 3.6785 | 0.1194 | 25.9524 |
| realized_value_quartile | Q4 | 101.0000 | 29.3169 | 0.1228 | 26.7421 |

## Grouped decision-specific permutation

| model | feature_group | baseline_value_capture_10 | permuted_mean_value_capture_10 | delta | permutation_standard_deviation | permutation_q025 | permutation_q975 |
|---|---|---|---|---|---|---|---|
| AP-EV | product | 0.2423 | 0.0922 | 0.1501 | 0.0206 | 0.0560 | 0.1286 |
| AP-EV | firmographics | 0.2423 | 0.2212 | 0.0211 | 0.0270 | 0.1704 | 0.2723 |
| AP-EV | account_history | 0.2423 | 0.2334 | 0.0088 | 0.0210 | 0.1967 | 0.2708 |
| AP-EV | sector_history | 0.2423 | 0.2346 | 0.0077 | 0.0109 | 0.2133 | 0.2535 |
| AP-EV | calendar | 0.2423 | 0.2423 | -0.0000 | 0.0000 | 0.2423 | 0.2423 |
| AP-EV | product_history | 0.2423 | 0.3196 | -0.0774 | 0.0163 | 0.2909 | 0.3498 |
| B4 XGBoost | product | 0.0770 | 0.0602 | 0.0168 | 0.0092 | 0.0444 | 0.0798 |
| B4 XGBoost | calendar | 0.0770 | 0.0770 | 0.0000 | 0.0000 | 0.0770 | 0.0770 |
| B4 XGBoost | firmographics | 0.0770 | 0.0797 | -0.0028 | 0.0186 | 0.0484 | 0.1138 |
| B4 XGBoost | sector_history | 0.0770 | 0.0845 | -0.0075 | 0.0096 | 0.0669 | 0.1030 |
| B4 XGBoost | account_history | 0.0770 | 0.1085 | -0.0315 | 0.0194 | 0.0734 | 0.1484 |
| B4 XGBoost | product_history | 0.0770 | 0.1648 | -0.0878 | 0.0169 | 0.1287 | 0.1930 |

`permutation_q025` and `permutation_q975` form the `permutation_distribution_interval_95`: empirical 2.5th and 97.5th percentiles across repeated perturbations. This interval describes permutation variability conditional on the fixed LOCKED cohort and frozen model. It is **not a sampling confidence interval**.

Correlated information across feature groups can attenuate grouped permutation importance. These results were not used for feature selection.

## B1 concentration context

The B1 product/sector concentration results are contextualized against the underlying LOCKED Maven opportunity mix. `selection_lift` is selection share divided by population share; value shares use observed realized value. This is concentration analysis, not fairness analysis.

Product HHI rises from `0.1675` in the LOCKED population to `0.8644` in B1's selected set, so B1 product concentration is substantially beyond the underlying product mix. Sector HHI rises more modestly, from `0.1100` to `0.1672`. Category-level lifts below show where those differences arise.

| dimension | population_hhi | selection_hhi |
|---|---|---|
| product | 0.1675 | 0.8644 |
| sector | 0.1100 | 0.1672 |

| dimension | category | population_share | selection_share | selection_lift | population_value_share | selected_value_share |
|---|---|---|---|---|---|---|
| product | GTK 500 | 0.0074 | 0.0732 | 9.8293 | 0.0550 | 0.1886 |
| product | GTX Plus Pro | 0.1290 | 0.9268 | 7.1829 | 0.2723 | 0.8114 |
| sector | software | 0.0868 | 0.1951 | 2.2467 | 0.1356 | 0.1842 |
| sector | services | 0.0546 | 0.1220 | 2.2339 | 0.0649 | 0.1193 |
| sector | telecommunications | 0.0596 | 0.0976 | 1.6382 | 0.0446 | 0.0675 |
| sector | technolgy | 0.1538 | 0.2195 | 1.4268 | 0.1280 | 0.1486 |
| sector | retail | 0.1687 | 0.2195 | 1.3009 | 0.1752 | 0.1842 |
| sector | entertainment | 0.0596 | 0.0732 | 1.2287 | 0.1068 | 0.1886 |
| sector | marketing | 0.0794 | 0.0488 | 0.6143 | 0.0653 | 0.0705 |
| sector | medical | 0.1464 | 0.0244 | 0.1666 | 0.1436 | 0.0371 |

## PDP / ICE

PDP/ICE was restricted to the five predefined inputs over empirical 5th–95th percentile grids with a deterministic 50-row ICE sample. Every result is **MODEL RESPONSE, NOT CAUSAL EFFECT**. Correlated-feature perturbations may generate unrealistic combinations.

| feature | mean_response | minimum | maximum |
|---|---|---|---|
| account_prior_opportunities | 0.0355 | 0.0352 | 0.0384 |
| account_prior_win_rate | 0.0355 | 0.0351 | 0.0366 |
| employees | 0.0360 | 0.0311 | 0.0377 |
| revenue | 0.0340 | 0.0275 | 0.0421 |
| sales_price | 0.0355 | 0.0346 | 0.0386 |

## Optional component SHAP

Timing multiclass and conditional-model SHAP were not collapsed into a single attribution because class/time-bin and augmented-time semantics would be ambiguous. Status: `NOT_EVALUABLE_METHOD_AMBIGUITY`. This did not block required GPU1 analyses.

## Scientific boundary

A1 remains `RETAIN_BASELINE`. All results are `POST_LOCK_DIAGNOSTIC_ONLY`, decision-ineligible, non-causal, and suitable only for prospective A2 hypotheses.
