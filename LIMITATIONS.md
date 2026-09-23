# Limitations

- Maven is a fictitious public CRM benchmark, not enterprise production evidence; its one-shot LOCKED cohort has 403 rows.
- No cold account appeared in Track-A LOCKED, so T2/G9 is not evaluable and AP-EV cannot be promoted.
- Track-A isotonic calibration improved Brier but its LOCKED slope was 0.244; calibration remains inadequate.
- Random Survival Forest was not run on Track A because no reliable implementation was installed; supported timing evidence is Cox, XGBoost AFT, and discrete terminal modeling.
- Criteo v2.1 is non-uniformly subsampled and anonymized; the causal bridge supports within-benchmark ranking comparisons only, not absolute-effect transportability.
- Olist features are sparse; AP-EV value/timing components collapsed to weak/tied rankings and `rho=0`.
- UCI is wholesale-heavy retail, not enterprise CRM; only 44,941 basket events with customer IDs were modeled.
- UCI unseen-customer RFM ValueCapture@10% (25.7%) is far below known-customer performance (60.7%).
- Cox emitted a convergence warning and yielded inverted/poor ranking (AUC 0.304); it is a `SCIENTIFIC_FAILURE` and not selectable.
- Fixed fusion's small capture advantage is uncertain; adaptive fusion's better AUC does not establish higher decision value.
- Rank-bootstrap P(top-K) is based on stored predictions, not full model retraining.
- Local batch serving and historical shadow replay are not production deployment or online validation.
- Causal D1 used constant propensity and was invalidated after balance diagnostics; D2 uses train-only nuisance models and held-out AIPW evaluation, with 0.075% of test propensities clipped.
- GPU peak memory for E28 could not be observed because WSL blocked NVML process-memory queries; training runtime and device identity were retained.
- Coverage is 19% over the whole research package and 72% over the focused stable scientific/control modules (contracts, evidence, paths, protection, scoring, and CLI); tests target high-risk invariants rather than training-loop line coverage.
