# ACCOUNT_PULSE_FINAL_STATUS

## Identity and environment

- Repository: `/mnt/c/Users/bjw-0/Downloads/AccountPulse`
- Git: valid repository on `main`; preserved tag `pre-track-a-r2-d2^{commit}` and pre-Track-A HEAD both `cb4233bf53ab2e6ddd6c26d86ab15631f17ea3`; nothing pushed
- Protocols: B/C `AP-V1-PROTOCOL-20260922-R2`; A `AP-V1-TRACKA-20260923-A1`; D `AP-V1-CAUSAL-BRIDGE-20260922-D2`
- Python: 3.12.14, Ubuntu-22.04 WSL, existing venv `$HOME/.local/share/accountpulse/project-venv`
- GPU: one NVIDIA GeForce RTX 4090 Laptop GPU; no multi-GPU claim
- CUDA: Torch 2.14.0+cu132 FP16 and XGBoost 3.4.1 CUDA qualification required
- Dependencies: `pyproject.toml` and `uv.lock`; existing CUDA stack preserved

## Track status

- Track A: `COMPLETE_A1_ONE_SHOT_LOCKED`, fictitious public CRM, 8,800 official opportunities; all five hashes exact
- Track B: `COMPLETE_R2`, unchanged
- Track C: `COMPLETE_R2`, unchanged
- Track D/E28: `COMPLETE_D2`, unchanged and separate causal leaderboard

## Track-A source, split, freeze, and LOCKED identity

- Source hashes: recorded in `SOURCE_MANIFEST.yaml` and `FREEZE_MANIFEST.yaml`
- Horizon: 60 days, selected from 90/60/30 using timestamp support and row minima only
- Split: FIT 1,670; VALIDATION 434; POLICY 273; LOCKED 403; three 60-day gaps
- Split SHA-256: `86933d0f680d046a4ceb8b182df5dbcf9c6abf2db536321836de1663f63da456`
- Freeze: `FROZEN`; frozen configuration uses discrete timing, conditional Platt win, XGBoost value, `H=60`, `rho=0.005`
- Receipt: `LOCKED_START_RECEIPT.json`; freeze hash `7a889b633b40b0dd73c2d392d773c409316250e7456da5089f9f4b1542ad5a33`
- LOCKED labels: 188 Won, 157 Lost, 58 open/censored at 60 days; opened once

## Track-A results and decision

- Strongest frozen business/overall baseline: B1 business heuristic
- B1 ValueCapture@10% / WinCapture@10%: 29.18% / 12.23%
- AP-EV ValueCapture@10% / WinCapture@10%: 24.23% / 12.77%
- AP-EV minus B1 value capture: −4.96 points; 10,000 account-bootstrap 95% CI [−14.45, 8.06]
- AP-EV minus B1 win capture: +0.53 points; 95% CI [−2.28, 5.46]
- LambdaMART ValueCapture@10%: 23.22%
- POSTASSIGN XGBoost ValueCapture@10%: 7.87%; association only, non-causal
- Calibration: isotonic Brier 0.3248 versus raw B4 0.4321, but slope 0.244
- Temporal robustness: recent-half gate passed
- Cold account: zero T2 rows; G9 `NOT_EVALUABLE`
- Gates: G0-G4 pass; G5 fail; G6 pass; G7 fail; G8 pass; G9 not evaluable; G10-G11 pass
- Decision: `RETAIN_BASELINE`

## Negative results and ablations

- AP-EV did not beat the simple heuristic; LambdaMART and POSTASSIGN also lost
- AP-EV locked ValueCapture@10%: full 24.23%; minus timing 20.99%; minus conditional value 9.68%; minus discount 25.32%; minus calibration 23.12%; minus account history 24.10%
- RSF: `NOT_RUN_NO_RELIABLE_INSTALLED_IMPLEMENTATION`
- Track-B sparse AP-EV and Track-C fusion negative results remain preserved
- Leakage diagnostic L1 reached 92.38% versus valid L0 14.34%; L1-L3 are ineligible

## Experiment and operations status

- E00-E35: registry updated; E30 frozen, E31 one-shot complete, E32 diagnostic complete; E28 D2 unchanged
- MLflow: SQLite tracking contains separate Track-A development and LOCKED runs; B/C/D evidence retained
- Batch serving: historical Maven replay serves the retained heuristic plus diagnostic fields; not production deployment
- Shadow replay: historical only; no online-deployment claim
- Canonical evidence: Parquet plus DuckDB, including Track-A ranking, calibration, bootstrap, robustness, ablation, and uncertainty tables

## Quality status

- Pytest: 32 passed
- Coverage: 19% whole research package; 72% focused stable scientific/control modules
- Ruff: pass
- Scoped strict mypy: pass, 10 source files
- `uv lock --check`: pass, 187 packages
- Build: wheel and sdist pass
- Installed CLI smoke: pass, 403-row Maven replay plus header
- Final CUDA smoke: pass; Torch FP16 finite, XGBoost CUDA predictions finite, one GPU, peak Torch allocation 54,657,024 bytes
- Raw data, model weights, MLflow database, caches, and secrets remain ignored; no push performed

## Claims

Supported: on this frozen fictitious public Maven benchmark, the business heuristic outperformed AP-EV on one-shot LOCKED ValueCapture@10%, so the baseline was retained. Track-D response/uplift divergence remains benchmark-specific causal evidence.

Prohibited: production deployment, enterprise revenue lift, causal effect from predictive ranking, AP-EV improvement, cold-account validation, or causal transportability across tracks.

## Remaining blocker/action

There is no external Maven acquisition blocker. The remaining Track-A limitation is scientific: no cold accounts occurred in LOCKED. A valid follow-up requires a genuinely future cohort and a new protocol identity; do not retune against A1 LOCKED. No command can repair that within A1.

## Handoff paths

- Technical report: `reports/AccountPulse_Technical_Report.md`
- Executive summary: `reports/AccountPulse_Executive_Summary.md`
- Resume evidence: `RESUME_EVIDENCE.md`
- Interview notes: `INTERVIEW_NOTES.md`
- Adversarial audit: `reports/adversarial_audit.md`
