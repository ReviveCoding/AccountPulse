# AccountPulse Cross-Track Synthesis

base_protocol: `AP-V1-TRACKA-20260923-A1`
diagnostic_protocol: `AP-V1-TRACKA-20260923-A1-DIAG1`
result_label: `POST_LOCK_DIAGNOSTIC_ONLY`
post_lock: `true`
decision_eligible: `false`


Evidence classes and estimands remain separate.

| Track | Evidence / estimand | Diagnostic synthesis |
|---|---|---|
| A Maven | `FICTITIOUS_PUBLIC`; predictive capacity ranking | B1 beat AP-EV on frozen ValueCapture@10%; complex timing/value composition did not overcome the price/category heuristic. |
| B Olist | Public marketplace; predictive seller/lead ranking | Sparse AP-EV did not beat XGBoost: estimated difference -7.06 points with intervals spanning zero. |
| C UCI | Public retail replay; predictive customer ranking | Representation/fusion improved or matched discrimination in places, but fixed late fusion added only 0.30 ValueCapture points over RFM with a CI spanning zero; simple RFM remained formidable. |
| D Criteo | Verified randomized advertising; causal treatment allocation | Uplift and response targeting differed materially at constrained capacity. This is causal allocation evidence, not propensity-ranking evidence and is not transportable to Maven. |

Calibration mattered most visibly in Track A, where the frozen isotonic mapping improved Brier versus raw B4 but transported with slope weakness. Across predictive tracks, better representation or discrimination did not guarantee greater decision value. Track D answers a different question: treatment-effect targeting rather than outcome propensity.
