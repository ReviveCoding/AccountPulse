# Interview Notes

## Business framing and estimands

AccountPulse asks who to prioritize under capacity, not who is caused to convert by outreach. Predictive outcome/timing/value, causal treatment effect, and sequential allocation remain separate. ValueCapture@10% is primary; WinCapture@10% is co-primary.

## Data and point-in-time design

Maven is fictitious public CRM and currently blocked at full-source acquisition. Olist permits only first-contact fields; seller linkage is outcome-only. UCI histories, item text, and graph edges are truncated strictly before each cutoff. R2 adds 90-day gaps so FIT labels mature before VALIDATION scoring and VALIDATION labels mature before TEST.

## Labels, leakage, and survival

Fixed horizons distinguish mature positives, mature negatives, and censored rows. Track-A Lost must be a competing terminal event, never ordinary censoring. Leakage variants L1-L3 exist only as ineligible diagnostics. Cox on UCI failed to converge/usefully rank and was retained as a negative result.

## Value, ranking, calibration, and capacity

Olist XGBoost ValueCapture@10% was 17.4%; sparse AP-EV was 10.4%. Isotonic was selected on validation, but no post-test retuning followed. On UCI, RFM captured 57.3%. Adaptive fusion achieved AUC 0.869, Brier 0.154, and slope 0.913; fixed fusion captured 57.5%, but its incremental CI crossed zero.

## Representation learning and ablations

Transformer improved discrimination over tabular models but lost on the primary capacity endpoint. GraphSAGE nearly matched RFM; text was weak. Removing tabular caused the largest adaptive-fusion capture drop. Removing graph or text did not materially hurt adaptive performance, so their incremental contribution is unsupported.

## Uncertainty, robustness, and cold entities

Inference resamples stored predictions by customer/lead; Olist also uses temporal-block sensitivity. UCI cold/unseen RFM capture was 25.7% versus 60.7% for known customers. The selective 0.8 P(top10) policy covered 9.7% and remained a challenger.

## GPU and MLOps

One GPU lock serializes heavy jobs. Torch FP16 and XGBoost CUDA were requalified after uv sync. Three Transformer seeds used BF16 AMP; GraphSAGE/text used FP32. MLflow uses SQLite; R2 version 3 of `accountpulse-fusion` is alias `challenger`, while Track-A logical families are registered as blocked. Batch scoring is historical UCI replay only. Shadow replay is not online deployment.

## Causal bridge

Criteo v2.1 came from Criteo's verified Hugging Face organization and passed the exact source
hash. With no business-time field, D2 uses a stable 60/20/20 hash split rather than inventing
chronology. Visit is primary and conversion secondary. Treatment is excluded from response
and T-learner features; the S-learner uses treatment only as the randomized intervention;
post-treatment exposure is excluded. A train-only propensity model and outcome nuisances feed
held-out AIPW evaluation. At 10% reach, the S-learner and response lists overlap 64.2%; uplift
adds 0.000933 visits per eligible user, CI [0.000633, 0.001255]. The advantage is supported at
5%/10%, not 20%/30%. This is causal evidence only within the released advertising benchmark.

## Negative results and next experiments

RFM remains the predictive decision baseline; AP-EV, Cox, text, Transformer, and adaptive fusion did not win the primary decision gate. Next: acquire the complete official Maven package, rerun G0-G4, pre-register/freeze Track-A LambdaMART/AP-EV and competing-risk comparisons, then execute one-shot LOCKED. Track D is complete and remains a separate causal leaderboard.
