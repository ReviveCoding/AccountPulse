# Adversarial Final Audit

## Corrected defects

- **Label-horizon overlap:** R1 FIT/VALIDATION outcomes extended past later scoring dates. R1 was invalidated; R2 added 90-day gaps and B/C models were rerun. Track A used a separate A1 60-day design and its LOCKED outcomes opened only after the freeze/receipt.
- **Tie ordering:** exact Olist EV/prior ties inherited chronological row order. R2 now uses an outcome-independent SHA-256 ID tie breaker.
- **Calibration API/metrics:** repaired scikit-learn's frozen-estimator API and prevented EV scores from entering probability metrics.
- **MLflow backend/package:** replaced retired file tracking with SQLite; version 1 is infrastructure-invalid, version 2 is R1-invalid, and R2 version 3 has minimal explicit requirements.
- **Causal D1 propensity assumption:** constant 0.85 propensity produced biased slice comparisons under observed covariate-linked assignment variation. D1 was retained as invalid diagnostic evidence; D2 uses train-only propensity/outcome nuisances and held-out AIPW evaluation. No D1 test result drove hyperparameter tuning.

## Passed controls

PIT availability invariant; three Track-A 60-day embargoes; Lost competing-event contract; future graph-edge exclusion; development-defined LambdaMART qids/bins; account-cluster bootstrap; fail-closed LOCKED access before receipt; exact five-file Maven hashes; deterministic tie breaking; CUDA-required stages; public/fictitious claim boundaries; no multi-GPU claim; treatment/exposure leakage exclusion; Track-D split disjointness; separate causal/predictive leaderboards; no causal transportability or production wording.

## Remaining limitations, not hidden

Track-A G5 and G7 failed; T2/G9 was not evaluable because no cold accounts appeared. RSF was not run because no reliable implementation was installed, and ordinary Harrell concordance is descriptive rather than the sole survival claim. Post-lock ablations cannot authorize retuning. UCI Cox remains a scientific failure. Track D is causal evidence only within Criteo. No result is production evidence. Git is valid and the pre-Track-A tag resolves to the preserved baseline commit.
