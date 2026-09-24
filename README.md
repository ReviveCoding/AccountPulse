# AccountPulse v1.0

Evidence-controlled predictive decision science for capacity-constrained prioritization. Preserved external-track protocol: `AP-V1-PROTOCOL-20260922-R2`; Track-A protocol: `AP-V1-TRACKA-20260923-A1`; separate causal bridge: `AP-V1-CAUSAL-BRIDGE-20260922-D2`. AccountPulse prediction is not itself causal outreach modeling.

## Current outcome

Track A completed on the official 8,800-opportunity Maven package with exact source hashes, a 60-day outcome horizon, three 60-day purge gaps, a real freeze, and one-shot LOCKED evaluation. AP-EV captured 24.2% of realized value at 10% capacity versus 29.2% for the frozen business heuristic (difference −5.0 points; 95% account-bootstrap CI [−14.5, 8.1]). It failed value and calibration gates; decision: **RETAIN_BASELINE**. Maven is fictitious public CRM evidence, not production evidence.

On UCI, RFM captured 57.3% of 90-day spend at 10% capacity. Fixed fusion captured 57.5%, but its paired 5,000-resample difference CI crossed zero; decision: **RETAIN_BASELINE**. Adaptive fusion led discrimination (AUC 0.869) but not value capture. On Olist, sparse AP-EV underperformed CUDA XGBoost (10.4% vs 17.4% ValueCapture@10%).

On Criteo, the response and validation-selected uplift top-decile lists overlapped 64.2%; held-out AIPW visit value favored uplift by 0.000933 per eligible user (95% CI [0.000633, 0.001255]). This causal result is benchmark-specific and does not enter or change the predictive leaderboard.

## Reproduce

The repository publishes source, manifests, and compact JSON results. Raw benchmark files,
derived Parquet evidence, model weights, and local runtime databases are excluded. Full
research reproduction requires acquiring the documented sources under their terms and a
qualified Ubuntu 22.04 WSL CUDA environment; GitHub Actions runs only the documented
source and control checks in [CI](.github/workflows/ci.yml).

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.local/share/accountpulse/project-venv"
uv sync --all-extras --locked
python -m accountpulse.pipeline run --config configs/full.yaml --dry-run
make verify
accountpulse score --as-of 2011-09-01 --capacity 0.10 --track uci-replay --output artifacts/replay_scores.csv
accountpulse score --as-of 2017-11-01 --capacity 0.10 --track maven-locked-replay --output artifacts/maven_replay_scores.csv
```

The sole local project environment is `$HOME/.local/share/accountpulse/project-venv`;
GPU stages fail closed without one visible CUDA GPU. See `BLOCKERS.md`,
`SOURCE_MANIFEST.yaml`, `SPLIT_MANIFEST.yaml`, and the reports under `reports/`.
The frozen scientific checkpoint is `accountpulse-v1.0-complete`. Subsequent slicing,
calibration and drift analysis, GPU TreeSHAP, grouped permutation, PDP/ICE, and AP-EV
component diagnostics are preserved separately on branch
`analysis/postlock-diagnostics` at tag `accountpulse-v1.0-diag1-gpu1`.
