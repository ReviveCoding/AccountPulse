# AccountPulse Executive Summary

AccountPulse is a governed predictive ranking system for allocating scarce sales/marketing attention. It does not estimate causal outreach lift.

The complete official Maven CRM benchmark passed all five expected hashes. Under protocol `AP-V1-TRACKA-20260923-A1`, timestamp support selected a 60-day outcome horizon and four chronological cohorts separated by 60-day purge gaps. All source, split, feature, model, calibration, ranking, gate, seed, code, and claim choices were frozen before the 403-row LOCKED cohort was opened once.

At 10% capacity, the frozen business heuristic captured 29.2% of realized value and 12.2% of wins. AccountPulse-EV captured 24.2% and 12.8%. Its value-capture difference was −5.0 percentage points with a 95% account-bootstrap interval of [−14.5, 8.1]. LambdaMART captured 23.2%. AP-EV failed the value and calibration gates, while cold-account robustness was not evaluable because no new account appeared. Decision: **RETAIN_BASELINE**.

The prior scientific state was preserved. On Olist R2, sparse AP-EV remained below CUDA XGBoost. On UCI R2, fusion did not materially beat RFM. The separate Criteo D2 causal bridge still shows that response and uplift lists differ at tight reach, but that causal result does not enter the predictive leaderboard or transport to CRM.

This is public/fictitious benchmark evidence, not production deployment evidence. Any Track-A redesign must receive a new protocol identity and genuinely new protected outcomes; A1 LOCKED cannot be reused for tuning.
