# Adversarial Final Audit

## Corrected defects

- **Label-horizon overlap:** R1 FIT/VALIDATION outcomes extended past later scoring dates. Classified as a scientific protocol defect; R1 external results were invalidated, protocol R2 added 90-day purge gaps, and all B/C models were rerun. Track-A LOCKED was never opened.
- **Tie ordering:** exact Olist EV/prior ties inherited chronological row order. R2 now uses an outcome-independent SHA-256 ID tie breaker.
- **Calibration API/metrics:** repaired scikit-learn's frozen-estimator API and prevented EV scores from entering probability metrics.
- **MLflow backend/package:** replaced retired file tracking with SQLite; version 1 is infrastructure-invalid, version 2 is R1-invalid, and R2 version 3 has minimal explicit requirements.
- **Causal D1 propensity assumption:** constant 0.85 propensity produced biased slice comparisons under observed covariate-linked assignment variation. D1 was retained as invalid diagnostic evidence; D2 uses train-only propensity/outcome nuisances and held-out AIPW evaluation. No D1 test result drove hyperparameter tuning.

## Passed controls

PIT availability invariant; label maturity and embargo; Lost competing-event contract; cold definition; future graph-edge exclusion; ranking qid validator; cluster-bootstrap unit; no LOCKED receipt before freeze; CUDA-required stage enforcement; official-source hashes; public/fictitious claim boundaries; deterministic capacity evaluation; no multi-GPU claim; treatment leakage exclusion; exposure exclusion; Track-D split disjointness; validation-only causal model selection; separate causal/predictive leaderboards; no causal transportability or production wording.

## Remaining limitations, not hidden

Track A full source and therefore leakage inflation, agent comparison, LambdaMART, full AP-EV ablations, competing risks, freeze, and LOCKED are blocked. UCI Cox is a scientific failure. Track D is causal evidence only within the released Criteo population and does not validate predictive transportability. No result is production evidence. The sandbox's read-only `.git` mount prevented repository initialization; the recovery command is in `BLOCKERS.md`.
