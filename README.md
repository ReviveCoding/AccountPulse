# AccountPulse v1.0

Evidence-controlled predictive decision science for capacity-constrained prioritization. Predictive protocol: `AP-V1-PROTOCOL-20260922-R2`; separate causal bridge: `AP-V1-CAUSAL-BRIDGE-20260922-D2`. AccountPulse prediction is not itself causal outreach modeling.

## Current outcome

Track A's complete official Maven CRM package is externally blocked, so no primary CRM freeze or LOCKED evaluation occurred. Track B (8,000 Olist leads) and Track C (824,364 customer-identified UCI transaction lines) ran under 90-day label embargoes. Track D completed on the verified 13.98M-row Criteo v2.1 randomized benchmark.

On UCI, RFM captured 57.3% of 90-day spend at 10% capacity. Fixed fusion captured 57.5%, but its paired 5,000-resample difference CI crossed zero; decision: **RETAIN_BASELINE**. Adaptive fusion led discrimination (AUC 0.869) but not value capture. On Olist, sparse AP-EV underperformed CUDA XGBoost (10.4% vs 17.4% ValueCapture@10%).

On Criteo, the response and validation-selected uplift top-decile lists overlapped 64.2%; held-out AIPW visit value favored uplift by 0.000933 per eligible user (95% CI [0.000633, 0.001255]). This causal result is benchmark-specific and does not enter or change the predictive leaderboard.

## Reproduce

```bash
source .accountpulse.env.sh
uv sync --all-extras --locked
python -m accountpulse.pipeline run --config configs/full.yaml --dry-run
make verify
accountpulse score --as-of 2011-09-01 --capacity 0.10 --track uci-replay --output artifacts/replay_scores.csv
```

The sole environment is `$HOME/.local/share/accountpulse/project-venv`; GPU stages fail closed without one visible CUDA GPU. Raw data is ignored. See `BLOCKERS.md`, `SOURCE_MANIFEST.yaml`, `SPLIT_MANIFEST.yaml`, and the reports under `reports/`.
