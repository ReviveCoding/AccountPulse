# Decision Memo

Decision: **RETAIN_BASELINE**.

The strongest decision-facing baseline is UCI RFM at 57.3% ValueCapture@10% and 21.3% WinCapture@10%. Fixed fusion is numerically higher on value (57.5%) and win capture (22.2%), but the value difference CI crosses zero and fixed fusion has a calibration slope of 1.63. Adaptive fusion improves AUC/calibration but not capture enough to displace RFM.

Do not promote sparse AP-EV: on Olist it captured 10.4% versus 17.4% for XGBoost. Do not freeze Track A until the complete official Maven files pass G0-G4. Do not interpret any ranking as causal treatment lift.

The separate randomized Criteo bridge found that high responders are not the same population
as high incremental responders: top-10% list overlap was 64.2%, and uplift targeting added
0.000933 AIPW incremental visits per eligible user versus response targeting (95% CI
[0.000633, 0.001255]). This does not alter the predictive `RETAIN_BASELINE` decision and does
not transport to AccountPulse datasets or enterprise outreach.
