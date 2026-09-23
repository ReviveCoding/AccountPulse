# Track A Post-Lock Diagnostics

base_protocol: `AP-V1-TRACKA-20260923-A1`
diagnostic_protocol: `AP-V1-TRACKA-20260923-A1-DIAG1`
result_label: `POST_LOCK_DIAGNOSTIC_ONLY`
post_lock: `true`
decision_eligible: `false`


This analysis reuses frozen A1 predictions/outcomes and is explanatory only. It does not retrain, retune, recalibrate, select a calibrator, change a gate, or supply promotion evidence. Maven is `FICTITIOUS_PUBLIC`. Predictive ranking is not a causal effect.

## Main explanation

B1 won at the frozen 10% capacity because its price-weighted product and sector priors concentrated selection on higher realized-value categories. AP-EV captured slightly more wins but less value, consistent with a decision-value miss rather than a simple discrimination failure. Selection-set composition and the B1 decomposition below quantify that mechanism.

| variant | value_capture_10 | win_capture_10 | selected_count |
|---|---|---|---|
| price_x_sector_prior | 0.2918 | 0.1223 | 41 |
| full_B1 | 0.2918 | 0.1223 | 41 |
| price_x_product_prior | 0.2505 | 0.1011 | 41 |
| sales_price_only | 0.2383 | 0.0957 | 41 |
| sector_prior_only | 0.1210 | 0.1064 | 41 |
| product_x_sector_prior | 0.0824 | 0.1064 | 41 |
| product_prior_only | 0.0018 | 0.0904 | 41 |

## B1 versus AP-EV disagreement

| group | rows | captured_value | win_rate |
|---|---|---|---|
| B1_AND_APEV | 17.0000 | 44749.0000 | 0.4706 |
| B1_ONLY | 24.0000 | 103539.0000 | 0.6250 |
| APEV_ONLY | 24.0000 | 78354.0000 | 0.6667 |
| NEITHER | 338.0000 | 281461.0000 | 0.4408 |

## Capacity behavior

| model | capacity | value_capture | win_capture | oracle_regret |
|---|---|---|---|---|
| B1 | 0.0500 | 0.1727 | 0.0638 | 56508.0000 |
| B1 | 0.1000 | 0.2918 | 0.1223 | 98303.0000 |
| B1 | 0.2000 | 0.5052 | 0.2394 | 157261.0000 |
| B1 | 0.5000 | 0.9111 | 0.5213 | 45155.0000 |
| AP-EV | 0.0500 | 0.1491 | 0.0798 | 68531.0000 |
| AP-EV | 0.1000 | 0.2423 | 0.1277 | 123488.0000 |
| AP-EV | 0.2000 | 0.4676 | 0.2606 | 176346.0000 |
| AP-EV | 0.5000 | 0.8733 | 0.5585 | 64395.0000 |
| LambdaMART | 0.0500 | 0.1323 | 0.0691 | 77072.0000 |
| LambdaMART | 0.1000 | 0.2322 | 0.1223 | 128631.0000 |
| LambdaMART | 0.2000 | 0.4610 | 0.2500 | 179704.0000 |
| LambdaMART | 0.5000 | 0.9198 | 0.5479 | 40753.0000 |
| B4 XGBoost | 0.0500 | 0.0625 | 0.0745 | 112541.0000 |
| B4 XGBoost | 0.1000 | 0.0770 | 0.1223 | 207481.0000 |
| B4 XGBoost | 0.2000 | 0.1824 | 0.2447 | 321283.0000 |
| B4 XGBoost | 0.5000 | 0.5060 | 0.5585 | 251019.0000 |
| D2 | 0.0500 | 0.1200 | 0.0585 | 83319.0000 |
| D2 | 0.1000 | 0.2353 | 0.1170 | 127046.0000 |
| D2 | 0.2000 | 0.4265 | 0.2234 | 197229.0000 |
| D2 | 0.5000 | 0.8672 | 0.5532 | 67479.0000 |
| D4 | 0.0500 | 0.1200 | 0.0585 | 83319.0000 |
| D4 | 0.1000 | 0.2353 | 0.1170 | 127046.0000 |
| D4 | 0.2000 | 0.4354 | 0.2287 | 192708.0000 |
| D4 | 0.5000 | 0.8672 | 0.5532 | 67479.0000 |

## Rank stability

| model | top10_jaccard_mean | spearman_mean | kendall_mean |
|---|---|---|---|
| B1 | 0.5744 | 0.9999 | 0.9961 |
| AP-EV | 0.5611 | 1.0000 | 0.9982 |
| LambdaMART | 0.5519 | 0.9999 | 0.9965 |

## Restricted worst-slice discovery

All rows below are `POST_HOC_SLICE_DISCOVERY`, depth <= 2, n >= 30, wins >= 5, and cannot alter A1.

| slice | slice_value | support_n | effect_estimate | ci_low | ci_high | bh_adjusted_p_value |
|---|---|---|---|---|---|---|
| product × company_size | ('MG Advanced', 'ENTERPRISE') | 34 | -0.1402 | -0.2612 | 0.1734 | 1.0000 |
| revenue_quartile | Q4 | 90 | -0.1190 | -0.3191 | 0.1773 | 1.0000 |
| company_size × revenue_quartile | ('ENTERPRISE', 'Q4') | 90 | -0.1190 | -0.3226 | 0.1866 | 1.0000 |
| revenue_quartile × account_history_bin | ('Q4', '6+') | 90 | -0.1190 | -0.3250 | 0.1793 | 1.0000 |
| revenue_quartile × region | ('Q4', 'United States') | 73 | -0.0916 | -0.3490 | 0.1843 | 1.0000 |
| company_size × region | ('LARGE', 'United States') | 77 | -0.0904 | -0.2232 | 0.1814 | 1.0000 |
| sector × account_history_bin | ('retail', '6+') | 68 | -0.0857 | -0.2670 | 0.0745 | 1.0000 |
| sector | retail | 68 | -0.0857 | -0.2601 | 0.1023 | 1.0000 |
| sector × region | ('retail', 'United States') | 50 | -0.0810 | -0.2827 | 0.1595 | 1.0000 |
| product × region | ('MG Advanced', 'United States') | 47 | -0.0795 | -0.1713 | 0.0923 | 1.0000 |
| product | MG Advanced | 61 | -0.0713 | -0.1935 | 0.0855 | 1.0000 |
| product × account_history_bin | ('MG Advanced', '6+') | 61 | -0.0713 | -0.1712 | 0.0953 | 1.0000 |

## AP-EV components and explainability

Stored frozen ablations are reported. Opportunity-level component inference, SHAP, grouped permutation, and PDP/ICE are `NOT_EVALUABLE_INFRASTRUCTURE_FAILURE` because CUDA availability was `False`; the required no-CPU-fallback control was enforced. PDP/ICE, if run under A2 with CUDA, must be labeled **MODEL RESPONSE, NOT CAUSAL EFFECT** and interpreted cautiously under correlated features.

## Decision

**A1 decision remains RETAIN_BASELINE.** Any response to these findings requires a genuinely new A2 protocol.

## GPU Infrastructure-Recovery Supplement

GPU capability recovered after DIAG1. Frozen B4 native TreeSHAP, exact AP-EV component reconstruction, grouped permutation, and bounded PDP/ICE were completed under `AP-V1-TRACKA-20260923-A1-DIAG1-GPU1`. This does not overwrite DIAG1's original infrastructure failure or change A1. Top LOCKED attribution groups were: account_history, calendar, product_history.
