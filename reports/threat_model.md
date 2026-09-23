# Scientific Threat Model

| Threat | Failure mode | Required control |
|---|---|---|
| Future leakage | post-score facts enter predictors | PIT contract and timestamp invariant |
| Target leakage | close/outcome-derived fields enter eligible model | explicit deny-list plus L1 diagnostic quarantine |
| Full-data target encoding | future labels affect encoding | fold/time-local encoding; L2 quarantine |
| Historical aggregate leakage | unresolved/future outcomes enter history | resolution time strictly before scoring; L3 quarantine |
| Repeated accounts | optimistic row-wise inference | chronological design and account-cluster bootstrap |
| Agent assignment bias | assignment proxies opportunity quality | PREASSIGN primary; POSTASSIGN associative only |
| Censoring | incomplete follow-up treated as failure | maturity states and survival methods |
| Competing risks | Lost treated as ordinary censoring | explicit Won/Lost terminal states |
| Immature labels | future positives mislabeled negative | full-horizon eligibility invariant |
| Temporal drift | historical fit fails in future | T1/recent cohorts, drift and calibration checks |
| Cold accounts/customers | entity memorization inflates results | T2/unseen-customer isolation |
| Class imbalance | misleading accuracy | PR metrics, calibration, capture |
| Heavy-tailed value | a few wins dominate | log/tail EDA, robust metrics, clustered bootstrap |
| Calibration error | scores misstate risk and EV | authorized Platt/isotonic comparison and reliability |
| Rank instability | fragile top-K actions | dependence-aware rank bootstrap and abstention analysis |
| Small samples | unstable slices/results | uncertainty intervals and minimum-support disclosure |
| Complexity | challenger overfits | bounded search, strong baselines, protected POLICY |
| Modality missingness | fusion fails on sparse/cold cases | missingness-aware gates and ablations |
| Schema drift | serving inputs violate training assumptions | schema contracts, unknown-category and missing-feature stress |
| Future graph edges | graph embeds post-boundary behavior | time-cut graph construction and tests |
| Invalid qids | LambdaMART learns artificial groups | temporal capacity-cohort contract and validation |
| Calibration leakage | LOCKED/target evidence selects calibrator | fit/validation/POLICY authorization controls |
| Locked-outcome tuning | one-shot evaluation becomes HPO | protected access controller, freeze and receipt |
| GPU fallback | system claim silently becomes CPU | CUDA-required checks and one-slot lock |
| Evidence inflation | public benchmark claimed as production | claim ledger and adversarial review |
