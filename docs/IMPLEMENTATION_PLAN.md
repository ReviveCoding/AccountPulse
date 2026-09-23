# Implementation Plan

Protocol identity: `AP-V1-PROTOCOL-20260922-R2` (R1 external-track results were invalidated after the adversarial audit identified missing label-horizon embargoes; Track-A LOCKED was never opened.)

This is an executable, resume-safe plan. Stage state is machine-readable in `artifacts/state/pipeline_state.json`; experiment evidence is authoritative in `EXPERIMENT_REGISTRY.yaml`.

1. Preserve and requalify the trusted runtime; declare dependencies without replacing the CUDA stack.
2. Acquire only canonical public sources, hash them, document provenance/licenses/claim boundaries, and isolate external blockers.
3. Validate schemas and build canonical DuckDB/Parquet data with reusable EDA artifacts.
4. Implement PIT feature contracts, forward-only aggregates, label maturity/competing risks, leakage challenges, chronological splits, cold-entity checks, and protected LOCKED access.
5. Train and calibrate strong business/classification/survival/value/ranking baselines; record all candidates in MLflow/evidence tables.
6. Develop AP-EV and uncertainty/abstention on authorized evidence; assess capacity, persona, robustness, and preregistered gates.
7. Rebuild comparable methods on Olist; implement UCI lifecycle tabular, temporal, graph, text, fusion, and ablation ladder as data/dependencies permit.
8. Keep the verified randomized Criteo causal bridge separate from predictive leaderboards.
9. Benchmark the single-GPU path and enforce its lock/no-fallback contract.
10. Freeze all scientific choices, issue a receipt, run one-shot LOCKED evaluation, then restrict work to diagnostics or a new protocol.
11. Register eligible model families, implement batch scoring and historical shadow replay, populate canonical evidence, and generate reports/claim-safe handoff.
12. Run adversarial audit plus pytest, coverage, ruff, mypy, build, clean-install and CLI verification.

Stages may be skipped only when their registry state records a precise external or scientific blocker. No blocked track stops independent work. Protected stages require successful predecessor gates and cannot be forced.
