# AccountPulse Repository Instructions

These rules govern every session and agent working in this repository.

## Runtime and compute

- Run all scientific Python work under Ubuntu-22.04 WSL.
- The Windows-backed source repository is `/mnt/c/Users/bjw-0/Downloads/AccountPulse`.
- The only primary project environment is `$HOME/.local/share/accountpulse/project-venv`.
- Set `UV_PROJECT_ENVIRONMENT` to that environment. Do not create or substitute another venv.
- Put large temporary files and caches under `$HOME/.cache/accountpulse`.
- Preserve the qualified CUDA Torch/XGBoost/CatBoost stack recorded in `artifacts/bootstrap/`.
- GPU-capable model stages must use CUDA and fail loudly if CUDA is unavailable. Never silently fall back to CPU.
- There is one physical GPU. Run at most one heavy GPU training process at a time and make no multi-GPU claims.
- CPU ETL, statistics, reporting, and tests may run concurrently when safe.

## Scientific integrity

- AccountPulse is predictive decision science, not a causal outreach model. Keep predictive, causal, and sequential-allocation estimands separate.
- Protected outcomes may not influence feature, model, hyperparameter, calibration, threshold, or claim choices.
- Never access LOCKED outcomes before a complete freeze. A scientific change after outcome exposure requires a new protocol identity.
- Predictive ranking must never be described as causal lift or treatment effect.
- Retain negative results, failed eligible candidates, and strong baselines. Do not tune away protected negative findings.
- Treat Maven CRM as `FICTITIOUS_PUBLIC`; Olist, UCI Online Retail II, and verified Criteo as their documented public evidence classes. Never upgrade public, fictitious, semi-synthetic, local, or replay evidence into enterprise-production claims.
- Enforce point-in-time availability, horizon maturity, competing-risk semantics, chronological splits, cold-entity separation, and dependence-aware inference.
- Leakage diagnostics L1-L3 are `INELIGIBLE_LEAKAGE_DIAGNOSTIC` and can never enter selection.
- Hindsight oracles are retrospective references only.

## Execution and evidence

- Use the pipeline state, experiment registry, source manifest, split manifest, freeze manifest, claim ledger, and canonical DuckDB/Parquet evidence to resume work.
- Classify failures as `INFRASTRUCTURE_FAILURE`, `SCIENTIFIC_FAILURE`, `EXTERNAL_BLOCKER`, or `EXPECTED_NEGATIVE_RESULT`.
- An external blocker in one track must not halt unrelated tracks.
- Final claims and resume bullets must be generated only from validated canonical evidence.
- Never use `force` to bypass protected scientific gates.
