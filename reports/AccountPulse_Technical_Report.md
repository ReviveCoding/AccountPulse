# AccountPulse v1.0 Technical Report

## Abstract

AccountPulse is an evidence-controlled predictive decision system for ranking entities under limited capacity. Preserved protocol `AP-V1-PROTOCOL-20260922-R2` covers Olist/UCI; `AP-V1-TRACKA-20260923-A1` covers the complete Maven CRM benchmark; separate causal protocol `AP-V1-CAUSAL-BRIDGE-20260922-D2` compares response and incremental targeting without merging leaderboards.

## Protocol correction

An adversarial audit found label-horizon overlap in R1. R2 inserted 90-day embargoes and reran all external-track models. These R2 results were preserved. Track A received a distinct A1 identity, was frozen before access, and opened its LOCKED outcomes exactly once.

## Track A design and result

The official package contains 8,800 opportunities and all five files matched the frozen acquisition hashes. A timestamp-only horizon qualification selected 60 days: 90 days could not support the preregistered partition minima, while 60 days supported FIT 1,670, VALIDATION 434, POLICY 273, and LOCKED 403 with three 60-day purge gaps. PREASSIGN excludes agent, manager, and team; outcome histories update only when `close_date < engage_date`.

Model choice used VALIDATION. POLICY selected XGBoost CUDA, isotonic fixed-horizon calibration, and an AP-EV composition with discrete timing, conditional Platt probability, XGBoost conditional value, `H=60`, and `rho=0.005`. The freeze bound source/data/model/code hashes, split identity, LambdaMART qids/FIT-only bins, gates, seeds, and claims before the receipt was created.

On 403 one-shot LOCKED opportunities, the frozen business heuristic achieved ValueCapture@10% 0.2918 and WinCapture@10% 0.1223. AP-EV achieved 0.2423 and 0.1277. The 10,000-resample account-cluster AP-EV-minus-baseline value difference was −0.0496, 95% CI [−0.1445, 0.0806]; win difference 0.0053, CI [−0.0228, 0.0546]. LambdaMART ValueCapture@10% was 0.2322 and POSTASSIGN XGBoost 0.0787. Isotonic reduced Brier from 0.4321 to 0.3248 but its slope was 0.244. G5 and G7 failed; G9 was not evaluable because all LOCKED accounts were already known. Decision: `RETAIN_BASELINE`.

Locked AP-EV ablations were: full 0.2423, minus timing 0.2099, minus conditional value 0.0968, minus discount 0.2532, minus calibration 0.2312, and minus account-history signal 0.2410. These are explanatory post-lock diagnostics, not tuning evidence. L1 future-field leakage reached 0.9238 validation ValueCapture@10% versus 0.1434 for valid L0, demonstrating the audit's sensitivity; L1-L3 remain ineligible.

## Sources and cohorts

Olist contains 8,000 leads. R2 uses 973 FIT, 1,132 VALIDATION, 1,278 TEST, and 4,617 purged leads. UCI contains 824,364 customer-identified lines, 44,941 baskets, and 70,782 snapshots: 40,481 FIT through March 2011, 4,998 June VALIDATION, 5,314 September TEST, and 19,989 purged snapshots.

## Methods

Only first-contact Olist features enter prediction. UCI models use RFM/tabular history, pre-cutoff sequences, PIT customer-product graphs, and cached descriptions for items already observed. Deep search is bounded; the Transformer finalist is replicated across three seeds. Gates are non-compensatory. Capacity curves cover 5/10/20/30%. Customer/lead cluster bootstrap uses 5,000 stored-prediction resamples.

## Track B results

CUDA XGBoost: AUC 0.568, ValueCapture@10% 17.4%, WinCapture@10% 19.7%. Sparse AP-EV: AUC 0.493, ValueCapture@10% 10.4%, WinCapture@10% 12.6%; `rho=0`. AP-EV minus XGBoost value-capture estimate -7.1%; lead-bootstrap CI [-28.5%, 12.2%], temporal-block CI [-11.5%, 8.0%]. Result: baseline retained.

## Track C results

RFM is the decision baseline: AUC 0.789, ValueCapture@10% 57.3%, WinCapture@10% 21.3%. XGBoost: 56.7% value capture. Transformer: AUC 0.856, value capture 53.9%. GraphSAGE: 56.4%. Text: 29.0%. Fixed fusion: 57.5%; adaptive fusion: AUC 0.869, Brier 0.154, value capture 57.4%.

Fixed fusion minus RFM = 0.30%, paired 95% CI [-0.99%, 1.92%]. This fails materiality/uncertainty support; decision `RETAIN_BASELINE`.

## Ablations and robustness

Removing tabular signal reduced adaptive capture to 53.9%. Removing graph (57.3%) or text (57.3%) did not show incremental benefit. Known-customer RFM capture was 60.7%; unseen-customer capture was 25.7%. Cold robustness is therefore a material limitation.

## Track D causal bridge

Criteo Uplift v2.1 was acquired from Criteo's verified Hugging Face organization at immutable
revision `2424920019e49d52d72c13ac1143ec5d53af276b`; its 13,979,592-row artifact matched the
preregistered SHA-256 exactly. Because no suitable business-time ordering exists, D2 uses a
stable disjoint row-hash split: 8,387,399 train, 2,797,932 validation, and 2,794,261 test.

Visit is primary and conversion secondary. The validation-selected S-learner is compared with
a treated-response ranker using held-out AIPW values from train-only propensity and outcome
nuisance models. Exposure is excluded. At 10% reach, list overlap is 64.2% and uplift minus
response value is 0.000933 incremental visits per eligible user (2,000-bootstrap 95% CI
[0.000633, 0.001255]). Corresponding differences are 0.001678 at 5% (CI [0.001380,
0.001994]), 0.000138 at 20% (CI includes zero), and 0.000001 at 30% (CI includes zero).
S-learner visit AUUC/Qini are 0.006837/0.003168 versus response 0.006669/0.003000.
Conversion results are secondary and nearly tied. Thus high responders and high incremental
responders are not the same population, especially at tight reach, but no effect is transported
to Maven, Olist, UCI, enterprise CRM, or production outreach.

The AIPW interpretation assumes consistency, no interference, positivity after documented
clipping, and conditional exchangeability supplied by the randomized experiments after
adjustment for the released covariates. Non-uniform subsampling limits absolute-effect claims.

The earlier D1 constant-propensity evaluation was invalidated after the balance audit detected
covariate-linked treatment-rate variation. D2 repaired the estimator without changing model
hyperparameters or using Track-B/C outcomes. The propensity nuisance AUC is 0.510 and 0.075%
of test predictions required clipping.

## Systems and operations

Post-sync Torch 2.14.0+cu132 and XGBoost 3.4.1 CUDA qualification passed on one RTX 4090 Laptop GPU. Track-A GPU training took 132.3 seconds and total development 149.5 seconds; NVML peak process memory was unavailable, while the final Torch smoke recorded 54.7 MB peak allocation. Transformer peak allocation was 138.6 MB per seed; GraphSAGE 237.1 MB; text MLP 21.0 MB. E28 CUDA training took 58.6 seconds and total execution 134.3 seconds; inference used host-array DMatrix fallback. MLflow tracks predictive and causal runs in SQLite. Historical replay rejects predictive promotion. Batch scoring is replay-only.

## Conclusion

The predictive scientific result remains disciplined retention across tracks: Track-A AP-EV failed value and calibration gates, sparse Olist AP-EV failed, and UCI fusion did not materially exceed RFM. The separate causal bridge shows response and incrementality rankings diverge at tight capacity. None of these public/fictitious results supports production or cross-domain causal claims.
