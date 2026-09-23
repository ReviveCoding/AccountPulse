# Track A Calibration and Drift

base_protocol: `AP-V1-TRACKA-20260923-A1`
diagnostic_protocol: `AP-V1-TRACKA-20260923-A1-DIAG1`
result_label: `POST_LOCK_DIAGNOSTIC_ONLY`
post_lock: `true`
decision_eligible: `false`


The frozen calibrator is unchanged; no new calibrator is selected.

## LOCKED calibration

| method | support_n | brier | log_loss | ece | calibration_in_the_large | calibration_slope |
|---|---|---|---|---|---|---|
| raw_B4 | 403 | 0.4321 | 1.5693 | 0.4309 | 3.2766 | 0.9140 |
| Platt_mapping | 403 | 0.3308 | 0.9079 | 0.2964 | 1.4746 | 1.7917 |
| isotonic_mapping | 403 | 0.3248 | 0.9169 | 0.2888 | 2.1894 | 0.2436 |
| frozen_selected_probability | 403 | 0.3248 | 0.9169 | 0.2888 | 2.1894 | 0.2436 |

The LOCKED calibration weakness is a level-and-slope transport failure: predictions understate the LOCKED win base rate and the selected isotonic mapping has a shallow slope. Isotonic calibration fitted on a small development sample has material step-function overfit risk. A2 should prospectively compare prespecified calibration methods on new validation data only.

## Largest measured drift diagnostics

| drift_type | field | metric | value |
|---|---|---|---|
| LABEL_OUTCOME_DRIFT | LOCKED | MEAN_REALIZED_VALUE | 1260.8015 |
| INPUT_DRIFT | employees | WASSERSTEIN | 461.0995 |
| INPUT_DRIFT | sales_price | WASSERSTEIN | 374.8750 |
| LABEL_OUTCOME_DRIFT | POLICY | MEAN_REALIZED_VALUE | 282.4396 |
| SCORE_DRIFT | B1 | WASSERSTEIN | 217.1514 |
| INPUT_DRIFT | account_prior_opportunities | WASSERSTEIN | 170.6345 |
| INPUT_DRIFT | revenue | WASSERSTEIN | 105.9217 |
| SCORE_DRIFT | AP-EV | WASSERSTEIN | 25.1801 |
| INPUT_DRIFT | account_prior_opportunities | PSI | 2.5340 |
| INPUT_DRIFT | company_age | WASSERSTEIN | 0.8681 |
| LABEL_OUTCOME_DRIFT | POLICY | OPEN_RATE | 0.7509 |
| INPUT_DRIFT | account_prior_win_rate | PSI | 0.6760 |
| LABEL_OUTCOME_DRIFT | LOCKED | WIN_RATE | 0.4665 |
| SCORE_DRIFT | B4 XGBoost | PSI | 0.3900 |
| LABEL_OUTCOME_DRIFT | LOCKED | LOST_RATE | 0.3896 |
| SCORE_DRIFT | LambdaMART | PSI | 0.3070 |
| INPUT_DRIFT | account_prior_win_rate | KS | 0.2785 |
| SCORE_DRIFT | AP-EV | PSI | 0.2559 |
| SCORE_DRIFT | B4 XGBoost | KS | 0.2412 |
| INPUT_DRIFT | sector | JENSEN_SHANNON | 0.2349 |

Input, score, and post-label outcome drift are reported separately. No diagnostic threshold is an A1 gate.
