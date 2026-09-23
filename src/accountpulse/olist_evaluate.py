"""Olist capacity, calibration, lead bootstrap, and temporal-block sensitivity."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from accountpulse.contracts import capacity_metrics
from accountpulse.evidence import EVIDENCE, write_table
from accountpulse.paths import REPO

SEED = 20260922
SCORES = {
    "business_prior": "prior_score",
    "logistic": "logistic_score",
    "xgboost_cuda": "xgboost_score",
    "simple_ev": "simple_ev_score",
    "sparse_accountpulse_ev": "accountpulse_ev_score",
}


def run(resamples: int = 5000) -> dict[str, object]:
    frame = pd.read_parquet(EVIDENCE / "olist_predictions.parquet")
    y = frame["converted"].to_numpy()
    value = frame["value_90d"].to_numpy(float)
    curves = []
    for model, column in SCORES.items():
        score = frame[column].to_numpy(float)
        for capacity in (0.05, 0.10, 0.20, 0.30):
            metric = capacity_metrics(score, value, y, capacity)
            curves.append({"track": "B", "model": model, **metric.__dict__})
    existing_curve_path = EVIDENCE / "capacity_curves.parquet"
    existing = (
        pd.read_parquet(existing_curve_path) if existing_curve_path.exists() else pd.DataFrame()
    )
    if not existing.empty:
        existing = existing.loc[existing["track"] != "B"]
    write_table("capacity_curves", pd.concat([existing, pd.DataFrame(curves)], ignore_index=True))

    raw = frame["xgboost_score"].to_numpy(float)
    method, platt, isotonic = joblib.load(
        REPO / "artifacts/models/olist/xgboost_calibration.joblib"
    )
    logit_raw = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
    candidates = {
        "raw": raw,
        "platt": platt.predict_proba(logit_raw.reshape(-1, 1))[:, 1],
        "isotonic": isotonic.predict(raw),
    }
    calibration = [
        {
            "track": "B",
            "model": f"xgboost_{name}",
            "selected_on_validation": name == method,
            "brier": brier_score_loss(y, probability),
            "log_loss": log_loss(y, np.clip(probability, 1e-7, 1 - 1e-7)),
            "roc_auc": roc_auc_score(y, probability),
            "top_decile_predicted": float(
                probability[probability >= np.quantile(probability, 0.9)].mean()
            ),
            "top_decile_observed": float(y[probability >= np.quantile(probability, 0.9)].mean()),
        }
        for name, probability in candidates.items()
    ]
    calibration_path = EVIDENCE / "calibration_metrics.parquet"
    old_calibration = (
        pd.read_parquet(calibration_path) if calibration_path.exists() else pd.DataFrame()
    )
    if not old_calibration.empty:
        old_calibration = old_calibration.loc[old_calibration["track"] != "B"]
    write_table(
        "calibration_metrics",
        pd.concat([old_calibration, pd.DataFrame(calibration)], ignore_index=True),
    )

    rng = np.random.default_rng(SEED)
    bootstrap_rows = []
    for iteration in range(resamples):
        indices = rng.integers(0, len(frame), len(frame))
        for model in ("xgboost_cuda", "sparse_accountpulse_ev"):
            metric = capacity_metrics(
                frame[SCORES[model]].to_numpy()[indices], value[indices], y[indices], 0.10
            )
            bootstrap_rows.append(
                {
                    "track": "B",
                    "iteration": iteration,
                    "model": model,
                    "cluster_unit": "lead",
                    "value_capture_10": metric.value_capture,
                    "win_capture_10": metric.win_capture,
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap_path = EVIDENCE / "bootstrap_results.parquet"
    old_bootstrap = pd.read_parquet(bootstrap_path) if bootstrap_path.exists() else pd.DataFrame()
    if not old_bootstrap.empty:
        old_bootstrap = old_bootstrap.loc[old_bootstrap["track"] != "B"]
    write_table("bootstrap_results", pd.concat([old_bootstrap, bootstrap], ignore_index=True))
    pivot = bootstrap.pivot(index="iteration", columns="model", values="value_capture_10")
    difference = pivot["sparse_accountpulse_ev"] - pivot["xgboost_cuda"]

    weeks = pd.to_datetime(frame["first_contact_date"], utc=True).dt.to_period("W").astype(str)
    unique_weeks = weeks.unique()
    block_differences = []
    for _ in range(1000):
        sampled_weeks = rng.choice(unique_weeks, len(unique_weeks), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(weeks.to_numpy() == week) for week in sampled_weeks]
        )
        apev = capacity_metrics(
            frame["accountpulse_ev_score"].to_numpy()[indices], value[indices], y[indices], 0.10
        )
        xgb = capacity_metrics(
            frame["xgboost_score"].to_numpy()[indices], value[indices], y[indices], 0.10
        )
        block_differences.append(apev.value_capture - xgb.value_capture)
    report = {
        "status": "COMPLETE",
        "selected_calibration": method,
        "apev_minus_xgboost_value_capture_10": {
            "estimate": float(
                capacity_metrics(
                    frame["accountpulse_ev_score"].to_numpy(), value, y, 0.10
                ).value_capture
                - capacity_metrics(frame["xgboost_score"].to_numpy(), value, y, 0.10).value_capture
            ),
            "lead_bootstrap_ci_95": [
                float(difference.quantile(0.025)),
                float(difference.quantile(0.975)),
            ],
            "temporal_block_ci_95": [
                float(np.quantile(block_differences, 0.025)),
                float(np.quantile(block_differences, 0.975)),
            ],
            "resamples": resamples,
        },
    }
    (EVIDENCE / "olist_evaluation_result.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
