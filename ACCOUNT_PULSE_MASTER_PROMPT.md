# AccountPulse v1.0 Master Scientific and Implementation Protocol

This file is the durable, semantically complete execution contract for AccountPulse v1.0. It supersedes conversation memory. Active protocol identity is `AP-V1-PROTOCOL-20260922-R2`; R2 added label-horizon embargoes after an adversarial audit invalidated external-track R1 splits, before any Track-A LOCKED access. Full title: **AccountPulse: Evidence-Controlled B2B Lead and Account Prioritization, Lifecycle Forecasting, and Capacity-Constrained Decision System**.

## Mission and estimands

The business decision is which leads or accounts should receive scarce Sales/Marketing capacity now to maximize future realized value while preserving win capture, calibration, temporal robustness, and decision reliability. The primary capacity is 10%; sensitivities are 5%, 20%, and 30%. The primary metric is ValueCapture@10% and co-primary is WinCapture@10%.

AccountPulse is predictive decision science. Keep three estimands distinct: (1) predictive—who converts, when, and at what value; (2) causal—whose behavior changes because of treatment; and (3) sequential allocation—how constrained resources evolve through time. A predictive ranking is never causal lift.

## Immutable environment policy

Ubuntu-22.04 WSL, repository `/mnt/c/Users/bjw-0/Downloads/AccountPulse`, and venv `$HOME/.local/share/accountpulse/project-venv` are already qualified. Caches belong in `$HOME/.cache/accountpulse`. Preserve the bootstrap evidence and CUDA-capable Torch, XGBoost, and CatBoost stack. Configure uv to the existing venv, declare compatible dependencies, retain the CUDA PyTorch source/backend, and rerun Torch and XGBoost CUDA smoke tests after synchronization. Dependency resolution must be corrected rather than accepting CPU fallback. Use exactly one heavy GPU process at a time; CPU work may overlap; make no multi-GPU claim. Record identity, precision, timings, throughput, peak VRAM, batch/sample/step counts, and relevant p50/p95 inference latency.

## Evidence tracks and claim boundaries

- **A—Maven Analytics CRM Sales Opportunities:** official/canonical files (`sales_pipeline.csv`, `accounts.csv`, `products.csv`, `sales_teams.csv`, `data_dictionary.csv`) searched locally first. Use for opportunity prioritization, terminal state/time, realized value, account/product/firmographic context, and ranking. Evidence is `FICTITIOUS_PUBLIC`. If official files require interactive login, mark `BLOCKED_EXTERNAL_DATA`, implement the full contract/tests, document exact recovery in `BLOCKERS.md`, and continue.
- **B—Olist Marketing Funnel + Brazilian E-Commerce:** real sparse funnel replication using only information available at `first_contact_date`; winner-only/post-close fields are forbidden as predictors; `seller_id` may construct post-boundary outcomes. Use fixed 90/180-day downstream marketplace value with maturity. If Kaggle credentials block acquisition, mark only that acquisition `BLOCKED_EXTERNAL_CREDENTIAL`.
- **C—official UCI Online Retail II:** `PUBLIC_REAL` wholesale-heavy retail transactions, not enterprise CRM. Use for repeat purchase, time to next purchase, activity/retention, future spend, sequences, description semantics, and customer-product graphs.
- **D—verified Criteo randomized uplift:** keep its causal leaderboard separate. Compare response/propensity and uplift targeting at 5/10/20/30%.

For every source record provider, official identity, version, license, acquisition method/time, filenames, sizes, SHA-256, shape, time range, limitations, and claim boundary in `SOURCE_MANIFEST.yaml` and `DATA_SOURCES.md`. Do not commit raw data when terms or size prohibit it.

## Desktop study and threat model

Before model development create a desktop study, business decision card, estimand contract, and threat model. Cover future/target leakage, full-data target encoding, repeated-account dependence, agent assignment bias, censoring, competing risks, immature labels, drift, cold entities, imbalance, heavy-tailed values, calibration and rank instability, small samples, complexity, missing modalities, and schema drift.

## Personas, point-in-time features, and leakage challenge

`PREASSIGN` is Track A's primary leaderboard and excludes assigned agent, manager, and sales team. `POSTASSIGN` is secondary and may use assignment known at scoring; salesperson effects are non-causal. Compare agent-blind and agent-aware formally.

`FEATURE_AVAILABILITY.csv` records feature, source, definition, event and availability timestamps, prediction timestamp, persona permissions, target-derived and historical-only flags, and reason. Enforce `availability_timestamp <= prediction_timestamp` with tests. Prefer deterministic audited PIT joins unless a stable feature store materially helps.

Create L0 valid PIT; L1 plus future-close fields; L2 plus full-data target encoding; L3 plus future account aggregates. L1-L3 are ineligible leakage diagnostics. Quantify inflation as QA only.

Forward-only account, product, sector, and (PostAssign only) agent history may include prior opportunities, wins/win rates, realized value statistics, terminal time, and recency. Any target-derived input uses only outcomes resolved strictly before scoring.

## Labels and splits

Track A terminal event time is `terminal_date - engage_date`; states are 0 open/censored, 1 Won, 2 Lost. Lost is a competing event, not censoring. Fixed Won horizons are 30/60/90/180 days and require full horizon observability; explicitly encode `MATURE_POSITIVE`, `MATURE_NEGATIVE`, `CENSORED`. Realized value is close value for Won and zero for Lost; analyze tails.

No random primary split. Track A is chronological by `engage_date`, approximately FIT 60%, VALIDATION 15%, POLICY 10%, LOCKED 15%; boundaries may use only availability, maturity, and counts. Freeze LOCKED without outcome inspection. Report T1 future (possibly known accounts) and T2 cold-account (absent from development). Track C uses chronological customer splits for known-customer future and unseen-customer evaluation. Future graph edges are forbidden.

## Modeling program

Do not weaken baselines. Business: B0 historical prior, B1 interpretable heuristic. Classification: B2 regularized Logistic, B3 CatBoost, B4 XGBoost CUDA (`device="cuda"`, `tree_method="hist"`). Timing/survival: B5 Cox PH, B6 Random Survival Forest, B7 XGBoost AFT, B8 competing-risk/discrete terminal. Value: B9 Gamma, B10 Tweedie, B11 XGBoost regression. Direct decisions: D0 P(Won), D1 predicted realized value, D2 P(Won)*E(Value|Won), D3 survival urgency, D4 discounted EV.

Implement CUDA LambdaMART with `rank:ndcg`; use realistic weekly/monthly qids chosen from development evidence and graded realized-value bins whose thresholds come only from authorized development/POLICY data.

Implement AccountPulse-EV:

`APEV_i(H) = sum_{t<=H} P(T_terminal=t|X_i) * P(Won|T_terminal=t,X_i) * E(Value|Won,T_terminal=t,X_i) * exp(-rho*t)`.

It must expose timing, win/loss, value, discount, and horizon. Tune H, rho, and component choices only on POLICY; freeze one primary configuration. Compare D0-D4, LambdaMART, and the strongest conventional baseline without assuming AP-EV wins.

Calibration candidates are raw, Platt/sigmoid, and isotonic on authorized evidence. Report Brier, log loss, intercept, slope, reliability, top-decile and horizon calibration; assess isotonic overfit. Survival reporting includes IPCW/Uno concordance, integrated Brier, time-dependent AUC and AUC@30/60/90, and calibration where supported—not just Harrell C. Preserve competing risks.

Capacity evaluation reports ValueCapture, WinCapture, NDCG, Precision/Recall@K, lift versus random and heuristic, value captured, and hindsight-oracle regret at 5/10/20/30%. Estimate rank/score distributions, median rank, P(top5/10/20) with dependence-aware bootstrap. A POLICY-selected challenger may auto-prioritize above a P(top-K) threshold and otherwise abstain/manual-review; report coverage/capture/uncertainty without assuming improvement.

Use account-cluster bootstrap for A, lead plus temporal-block sensitivity for B, and customer-cluster bootstrap for C—about 5,000 development and 10,000 final primary resamples when reasonable, normally over stored predictions. Correct multiplicity for preregistered secondary families where appropriate.

## External and representation tracks

Olist retrains comparable priors, Logistic, CatBoost/XGBoost, simple EV, and sparse AP-EV on Olist-valid features; never directly transfer Maven models. Test whether timing/value awareness improves decisions over propensity-only scoring.

UCI aggregates lines to basket events with timestamp, spend, item/product counts, average price, recency, cancellation, and semantic summary. Tasks: next purchase 30/90d, time to next purchase, and future 90d spend. Ladder: G0 RFM+Logistic, G1 XGBoost, G2 survival, G3 Temporal Transformer, G4 GraphSAGE, G5 text embedding+MLP, G6 early fusion, P2 adaptive AccountPulse-Fusion.

Transformer bounded search: layers 2/3/4, d_model 128/256, heads 4/8, dropout .1/.2, history 20/50/100; BF16/AMP where safe, early stop, clipping, pinned/prefetched data, fixed seeds; three seeds for the finalist. GraphSAGE uses PIT-correct customer-product bipartite graphs with count, quantity, spend, and recency-weighted signals, never future edges. Cache immutable GPU SentenceTransformer item embeddings. Compare early concatenation, fixed late fusion, and adaptive gated fusion across tabular/temporal/graph/text, with gating from history length, missingness, cold status, and tabular state.

Mandatory fusion ablations remove temporal, graph, text, tabular, and adaptive gate; AP-EV ablations remove timing, value, discount, calibration, and historical priors. Retain all negative results.

## Robustness, freeze, gates, and protected evaluation

Track A slices: recent cohort, cold account, sector, company size, revenue quantile, region, product, agent, manager, and high-value tail. Track C: missing text, truncated history, cancellation-heavy, country, cold customer, unknown item/category. Stress missing features, unknown categories, schema shift, and truncated history; report discrimination, calibration, capture, and rank stability.

Gate hierarchy is non-compensatory: G0 source integrity, G1 schema, G2 zero PIT leakage, G3 maturity/censoring, G4 valid predictions, G5 ValueCapture, G6 WinCapture noninferiority, G7 calibration, G8 temporal robustness, G9 cold robustness, G10 systems validity, G11 claim audit. Decisions: `PROMOTE_AP_EV`, `PROMOTE_SEGMENT_ONLY`, `RETAIN_BASELINE`, `RESEARCH_ONLY`. Never hide a failed gate with a weighted score.

Preregister H1 AP-EV material ValueCapture improvement, H2 WinCapture noninferiority, H3 no material calibration regression, H4 no catastrophic recent-cohort degradation, H5 acceptable cold performance. Choose margins before LOCKED and store them in `configs/gates.yaml` and `FREEZE_MANIFEST.yaml`.

Before accessing LOCKED outcomes freeze source/data hashes, split IDs, features, candidates/artifacts/hyperparameters/calibrators, ranking definitions, capacity, margins, bootstrap, seeds, evaluation code identity, gates, and claim templates. Write `FREEZE_MANIFEST.yaml` and `LOCKED_START_RECEIPT.json`. Evaluation is one-shot. Evidence-preserving infrastructure recovery is allowed; scientific retuning after exposure requires a new protocol.

## MLOps, serving, monitoring, and evidence

Track major runs in MLflow with source/git identity, hashes, split and feature contracts, configuration, seed, metrics, plots, runtime/GPU evidence, artifacts, and protocol. Register logical families `accountpulse-win`, `accountpulse-terminal`, `accountpulse-value`, `accountpulse-ranker`, and `accountpulse-fusion`; aliases champion/challenger/shadow follow gates.

Primary serving is batch: `accountpulse score --as-of DATE --capacity 0.10`, returning opportunity/account IDs, win probability, expected terminal days/value, AP-EV score, rank, p_top10, review flag, and model version. Any FastAPI is local only, not production.

Monitor immediate drift, missing/unknown inputs, score/rank drift, freshness, latency and errors; after maturity monitor wins, calibration, capture, survival, and segments. Historical shadow replay must score, await historically mature outcome, evaluate, detect drift, train/validate challenger, shadow, and promote/reject. Call it replay, not online deployment.

DuckDB is the analytical layer over canonical compact Parquet tables: runs, predictions, survival/value predictions, ranking/calibration/capacity/bootstrap/robustness/systems metrics, experiment status, and claim ledger. Final numbers come only from validated evidence.

## Software, pipeline, reports, and completion

Use ruff, scoped strict mypy, pytest/coverage, package build, clean-install/package and CLI smokes. Meaningful tests cover PIT/history/target leakage, maturity, competing risks, split/cold separation, graph future leakage, ranking qids, capacity, calibration, bootstrap units, freeze/locked controls, claim safety, and CUDA enforcement.

Provide `make bootstrap acquire qualify eda develop deep causal freeze locked systems mlops report verify all` and `python -m accountpulse.pipeline run --config configs/full.yaml`, supporting resume, stage selection, dry-run, structured status, safe retries, and protected state. `force` never bypasses gates.

Experiment IDs E00-E35 and their required questions are authoritative in `EXPERIMENT_REGISTRY.yaml`: desktop, acquisition, schema, EDA, labels, feature availability, leakage, splits, business baselines, classification, calibration, survival, competing risk, value, simple EV, LambdaMART, AP-EV, uncertainty/abstention, capacity curves, agent comparison, robustness, Olist, UCI baselines, Transformer, GraphSAGE, text, fusion, ablations, causal bridge, GPU systems, freeze, locked final, post-lock diagnostics, registry, shadow replay, final report. Use smoke -> bounded search -> 2-3 finalists -> replication -> freeze; preserve every candidate.

Mandatory documents are README, DATA_CARD, MODEL_CARD, DECISION_MEMO, LIMITATIONS, RESUME_EVIDENCE, INTERVIEW_NOTES, BLOCKERS, FEATURE_AVAILABILITY, SOURCE/SPLIT/EXPERIMENT/FREEZE/CLAIM manifests and technical/executive reports. Resume evidence contains only supported results and 2-4 safe bullets; claims list evidence, safe wording, and prohibited stronger wording. Interview notes cover the complete scientific and system design.

Before completion perform an adversarial audit for all leakage/timing/maturity/censoring/competing-risk/cold/graph/ranking/bootstrap/calibration/baseline/locked/GPU/evidence/claim/reproducibility failures. Fix non-protected defects; use a new protocol if protected science must change. Classify failures as infrastructure, scientific, external, or expected negative.

Completion means all feasible stages, evidence, tests, reports, registry, serving, replay, freeze/one-shot locked evaluation, and structured handoff are complete—not merely scaffolded or coded. Final output is headed `ACCOUNT_PULSE_FINAL_STATUS` and includes environment identities, all track and E00-E35 states, source/split/freeze/locked identities, baseline/challenger results, primary metrics and robustness, negative/ablation/external/causal conclusions, GPU/MLflow/replay/tooling status, report paths, safe/prohibited claims, and every blocker with exact recovery action.
