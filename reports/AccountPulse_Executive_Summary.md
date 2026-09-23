# AccountPulse Executive Summary

AccountPulse built a governed ranking system for deciding which entities receive scarce attention, while explicitly separating prediction from causal outreach impact.

The primary CRM study could not be run because Maven's complete official dataset requires an external interactive download; protected outcomes were never opened and no CRM lift claim is made. Two independent public tracks were completed under corrected 90-day maturity embargoes.

- Olist: CUDA XGBoost captured 17.4% of 90-day seller value at 10% capacity; sparse AP-EV captured 10.4%. Baseline retained.
- UCI: RFM captured 57.3%; fixed fusion captured 57.5%, but its +0.3% difference had a 95% CI crossing zero. Adaptive fusion improved AUC to 0.869 but not the decision metric enough to promote.

A separate randomized Criteo causal bridge used a verified 13.98M-row source and held-out AIPW
evaluation. At 10% reach, the response and uplift lists overlapped only 64.2%; the selected
S-learner added 0.000933 incremental visits per eligible user (95% CI [0.000633, 0.001255]).
The advantage was supported at 5% and 10%, but not 20% or 30%. This result is confined to the
released advertising benchmark and is not part of the predictive leaderboard.

Predictive decision: **RETAIN_BASELINE**, unchanged. The most important next step is acquiring the complete official CRM source, qualifying it, freezing the protocol, and running the one-shot locked evaluation—not further tuning public benchmarks.
