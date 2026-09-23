# Track A Operational Monitoring Specification

base_protocol: `AP-V1-TRACKA-20260923-A1`
diagnostic_protocol: `AP-V1-TRACKA-20260923-A1-DIAG1`
result_label: `POST_LOCK_DIAGNOSTIC_ONLY`
post_lock: `true`
decision_eligible: `false`


This is a local production-style specification, **not a deployed monitor**.

## Immediate unlabeled monitoring

- Data quality: schema, missingness, ranges, category validity, identifier uniqueness, batch volume, and point-in-time availability.
- Input drift: numeric KS/Wasserstein/PSI and categorical Jensen-Shannon/frequency/unseen-category diagnostics.
- Score drift: B1, AP-EV, B4, and LambdaMART distribution and rank-shift diagnostics.
- Attribution drift: normalized grouped attribution ranks when CUDA inference is available.
- Selection concentration: account/sector/product/region HHI and top-category share. This is concentration analysis, not fairness analysis.

## Delayed labeled monitoring

- After 60-day maturity: ValueCapture, WinCapture, precision/recall, calibration, oracle regret, and slice degradation.
- Preserve open/censored semantics and account-cluster uncertainty.
- Revisit only under a new protocol; never tune from delayed A1 outcomes.

## PROPOSED A2 triggers

All thresholds are **PROPOSED, NOT A1-VALIDATED**: PSI >= 0.20; categorical Jensen-Shannon >= 0.10; calibration ECE >= 0.10; ValueCapture loss >= 5 percentage points; selection HHI increase >= 0.10; or a sufficiently supported slice whose adjusted degradation interval excludes zero. These require prospective validation.

## GPU Infrastructure-Recovery Supplement

The completed GPU1 attribution baseline enables the following **PROPOSED A2, NOT A1-VALIDATED** monitoring: feature-attribution rank drift, normalized mean-|SHAP| drift, top-feature turnover, and grouped attribution drift. Monitor both encoded and raw/grouped levels, compare like-for-like frozen preprocessing, and investigate rather than automatically promote or retrain. None is a retroactive A1 gate.
