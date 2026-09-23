# Decision Memo

Decision: **RETAIN_BASELINE**.

For primary Track A, retain the frozen business heuristic. On 403 one-shot LOCKED opportunities it captured 29.2% of 60-day realized value at 10% capacity versus 24.2% for AP-EV. The AP-EV difference was −5.0 points with 95% account-cluster CI [−14.5, 8.1]. WinCapture was non-inferior (12.8% versus 12.2%), but G5 and G7 failed and G9 was not evaluable. LambdaMART captured 23.2%; POSTASSIGN XGBoost captured 7.9% and is non-causal.

Track-B/C R2 remain unchanged: do not promote sparse Olist AP-EV, and retain UCI RFM over fusion. Do not interpret any predictive ranking as causal treatment lift.

The separate randomized Criteo bridge found that high responders are not the same population
as high incremental responders: top-10% list overlap was 64.2%, and uplift targeting added
0.000933 AIPW incremental visits per eligible user versus response targeting (95% CI
[0.000633, 0.001255]). This does not alter the predictive `RETAIN_BASELINE` decision and does
not transport to AccountPulse datasets or enterprise outreach.
