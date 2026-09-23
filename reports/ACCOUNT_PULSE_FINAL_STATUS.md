# ACCOUNT_PULSE_FINAL_STATUS

## Identity and environment

- Repository: `/mnt/c/Users/bjw-0/Downloads/AccountPulse`
- Predictive protocol: `AP-V1-PROTOCOL-20260922-R2`; causal bridge protocol: `AP-V1-CAUSAL-BRIDGE-20260922-D2`
- Git/source identity: `git init -b main` was attempted, but the sandbox mounts `.git/info/` read-only; source-tree digest and exact recovery are recorded under `artifacts/state/` and `BLOCKERS.md`. Nothing was pushed.
- Python: 3.12.14, Ubuntu-22.04 WSL, venv `$HOME/.local/share/accountpulse/project-venv`
- GPU: one NVIDIA GeForce RTX 4090 Laptop GPU; no multi-GPU claim
- CUDA: Torch 2.14.0+cu132, CUDA runtime 13.2; FP16 matmul passed
- XGBoost: 3.4.1, `device="cuda"`, `tree_method="hist"`; final CUDA qualification passed
- Dependencies: `pyproject.toml` + `uv.lock`, 187 resolved packages, lock check passed; no second venv

## Evidence tracks

- Track A: `BLOCKED_EXTERNAL_DATA`. Official 499-row Maven preview supports schema smoke only; full primary CRM package is absent. Claim boundary `FICTITIOUS_PUBLIC_PREVIEW_ONLY`.
- Track B: `COMPLETE_R2`. 8,000 Olist leads, first-contact predictors only, 90-day outcome, label embargo, chronological test.
- Track C: `COMPLETE_R2`. 824,364 customer-identified lines, 44,941 baskets, 70,782 lifecycle snapshots, label embargo and known/unseen evaluation.
- Track D: `COMPLETE_D2`. Verified official Criteo Hugging Face source, exact expected hash, 13,979,592 rows; separate randomized causal benchmark.

## Experiment program

| IDs | Status |
|---|---|
| E00 | COMPLETE |
| E01 | PARTIAL_EXTERNAL_BLOCKERS |
| E02-E04 | PARTIAL_COMPLETE_B_C |
| E05 | COMPLETE |
| E06 | BLOCKED_TRACK_A |
| E07 | PARTIAL_COMPLETE_B_C |
| E08-E09 | BLOCKED_TRACK_A |
| E10 | PARTIAL_COMPLETE_B_C |
| E11 | PARTIAL_SCIENTIFIC_FAILURE_C |
| E12 | BLOCKED_TRACK_A |
| E13-E14 | PARTIAL_COMPLETE_B |
| E15 | BLOCKED_TRACK_A |
| E16 | PARTIAL_COMPLETE_SPARSE_B_NEGATIVE |
| E17 | PARTIAL_COMPLETE_C |
| E18 | COMPLETE_B_C |
| E19 | BLOCKED_TRACK_A |
| E20 | PARTIAL_COMPLETE_C |
| E21 | COMPLETE_NEGATIVE_RESULT |
| E22 | COMPLETE |
| E23-E25 | COMPLETE_NEGATIVE_RESULT |
| E26 | COMPLETE_NONINFERENTIAL_CHALLENGER |
| E27 | PARTIAL_FUSION_COMPLETE_APEV_BLOCKED |
| E28 | COMPLETE_D2 |
| E29 | COMPLETE |
| E30 | BLOCKED_TRACK_A_G0 |
| E31 | BLOCKED_NOT_OPENED |
| E32 | NOT_APPLICABLE_LOCKED_NOT_OPENED |
| E33 | PARTIAL_COMPLETE |
| E34 | COMPLETE_HISTORICAL_REPLAY |
| E35 | COMPLETE |

Detailed per-experiment questions, artifacts, gates, seeds, and claim boundaries are in `EXPERIMENT_REGISTRY.yaml`.

## Scientific identities and decisions

- Source identity: hashes, sizes, shapes, dates, licenses, and limitations in `SOURCE_MANIFEST.yaml`.
- Split identity: R2 90-day embargoed partitions in `SPLIT_MANIFEST.yaml`.
- Freeze identity: `NOT_FROZEN_BLOCKED_EXTERNAL_DATA` in `FREEZE_MANIFEST.yaml`.
- LOCKED identity: none; `LOCKED_START_RECEIPT.json` intentionally does not exist and LOCKED outcomes were not opened.
- Final promotion decision: `RETAIN_BASELINE`.

## Results

- Strongest business baseline: UCI RFM, ValueCapture@10% 57.25%, WinCapture@10% 21.27%, AUC 0.789.
- Strongest conventional ML discrimination: UCI CUDA XGBoost AUC 0.791; ValueCapture@10% 56.72%.
- LambdaMART: not run—scientifically gated by missing complete Track-A source.
- AccountPulse-EV: sparse Olist version captured 10.35% of value versus 17.41% for CUDA XGBoost; AP-EV minus XGBoost estimate −7.06 points, lead-bootstrap 95% CI [−28.46, +12.21]. `rho=0`; baseline retained.
- AccountPulse-Fusion: fixed fusion ValueCapture@10% 57.55%; adaptive fusion AUC 0.869, Brier 0.154, calibration slope 0.913, ValueCapture@10% 57.37%.
- Fixed fusion minus RFM ValueCapture@10%: +0.30 points, customer-bootstrap 95% CI [−0.99, +1.92]; not materially supported.
- Temporal Transformer ensemble: AUC 0.856, Brier 0.155, ValueCapture@10% 53.91%.
- GraphSAGE: AUC 0.765, ValueCapture@10% 56.36%.
- Text embedding + MLP: AUC 0.739, ValueCapture@10% 28.99%.
- Cox: convergence warning, AUC 0.304, ValueCapture@10% 4.03%; `SCIENTIFIC_FAILURE`, ineligible.

## E28 causal bridge — separate leaderboard

- Source: Criteo Uplift v2.1 from verified Criteo Hugging Face organization, revision `2424920019e49d52d72c13ac1143ec5d53af276b`, SHA-256 `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`, CC BY-NC-SA 4.0.
- Hash split: TRAIN 8,387,399; VALIDATION 2,797,932; TEST 2,794,261. It is intentionally not chronological because no suitable business-time ordering exists.
- Validation-selected uplift method: S-learner. Primary outcome: visit; conversion secondary.
- Visit uplift-minus-response AIPW value per eligible user: 5% `0.001678` (95% CI `[0.001380, 0.001994]`); 10% `0.000933` (`[0.000633, 0.001255]`); 20% `0.000138` (`[-0.000022, 0.000299]`); 30% `0.000001` (`[-0.000103, 0.000106]`).
- Matched-list overlap: 52.8%, 64.2%, 86.3%, and 87.1% at 5%, 10%, 20%, and 30% reach.
- Visit AUUC/Qini: S-learner `0.006837/0.003168`; response `0.006669/0.003000`. Conversion AUUC was nearly tied: uplift `0.000877`, response `0.000885`.
- Conclusion: selecting likely responders does not identify the same population as selecting likely incremental responders, especially at tight capacity. Uplift's advantage is supported at 5%/10%, not 20%/30%.
- D1 constant-propensity evaluation was invalidated after balance audit; D2 uses train-only propensity/outcome nuisances with held-out AIPW. This does not alter predictive R2 or establish transportability.

## Robustness, calibration, and negative findings

- Known-customer RFM ValueCapture@10%: 60.73%; unseen-customer: 25.67%. Cold robustness is inadequate for promotion.
- Adaptive fusion top-decile predicted/observed rates: 92.14% / 92.67%.
- Fixed fusion was slightly stronger on value capture but calibration slope was 1.63.
- Removing tabular reduced adaptive capture to 53.91%; removing graph/text did not establish incremental benefit.
- Transformer improved discrimination but not primary capture. Graph nearly matched RFM. Text was weak. Selective ranking remained a challenger.
- R1 external-track results were invalidated after adversarial audit found label-horizon overlap; R2 added purge gaps and reran all B/C models. Track-A LOCKED was never exposed.

## Systems and MLOps

- GPU evidence: `artifacts/bootstrap/final_gpu_smoke.json`; FP16 Torch and CUDA XGBoost passed after final locked dependency sync.
- Three Transformer seeds used BF16 AMP, 512 batch size, 138.6 MB peak allocated VRAM, 126-154 seconds/seed.
- GraphSAGE: FP32, 237.1 MB peak, 9.95 seconds. Text MLP: FP32, 21.0 MB peak, 42.2 seconds.
- MLflow: SQLite tracking database at `artifacts/evidence/mlflow.db`; five logical families registered. `accountpulse-fusion` R2 version 3 has alias `challenger`; no champion promotion.
- MLflow causal evidence: D1 diagnostic and completed D2 runs are tracked under experiment `accountpulse-causal-bridge`.
- Historical shadow replay: complete, decision `RETAIN_BASELINE`; explicitly not online deployment.
- Batch serving: installed CLI smoke passed from `/tmp`; output contains 5,314 replay scores. This is local historical UCI replay, not production serving.
- E28 systems: CUDA XGBoost training 58.6 seconds, total 134.3 seconds, 13,979,592 samples. WSL blocked NVML process-memory reporting; host-array inference fallback is disclosed.

## Software verification

- pytest: 24 passed
- coverage: 21% over the whole research package; 47% over the focused stable scientific/control module set
- ruff: passed
- mypy: passed, strict scope over 10 stable scientific/control modules
- package build: wheel and sdist passed
- installed-package smoke: passed in the mandated venv, no second environment
- CLI smoke: passed
- uv lock check: passed

## Deliverables

- Technical report: `reports/AccountPulse_Technical_Report.md`
- Executive summary: `reports/AccountPulse_Executive_Summary.md`
- Resume evidence: `RESUME_EVIDENCE.md`
- Interview notes: `INTERVIEW_NOTES.md`
- Adversarial audit: `reports/adversarial_audit.md`
- Canonical evidence: `artifacts/evidence/accountpulse.duckdb` and Parquet tables

## Supported and prohibited claims

Supported wording is enumerated in `CLAIM_LEDGER.yaml`. Safe headline: a reproducible public-benchmark decision pipeline was built; disciplined evaluation retained strong baselines after AP-EV/fusion challengers failed materiality or decision gates.

Prohibited: production deployment, enterprise CRM lift, cross-domain causal outreach impact, salesperson causal effects, successful Track-A LOCKED evaluation, AP-EV superiority, fusion material improvement, causal transportability, or multi-GPU performance.

## Remaining blockers and exact recovery

1. **Track A full Maven CRM — external.** The official public page provides only a 499-row preview; primary science requires the complete official package. Complete the browser download at `https://mavenanalytics.io/data-playground/crm-sales-opportunities`, place the five unmodified CSVs in `data/incoming/maven/`, then run `make acquire && make develop`. After G0-G4 pass, create a real freeze before `make locked`.
2. **Track-A E06/E08-E20/E30-E32 — scientifically gated.** These cannot be made valid with preview data. Recovery is the Track-A acquisition action above, followed by source/schema/PIT/maturity gates, freeze, and one-shot LOCKED execution.
3. **Git initialization — sandbox filesystem.** `.git/info/` is read-only here. After the session run: `cd /mnt/c/Users/bjw-0/Downloads/AccountPulse && git init -b main`. Do not publish or push unless separately intended.
