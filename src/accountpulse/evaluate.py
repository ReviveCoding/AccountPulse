"""Capacity curves, cluster bootstrap, calibration, slices, and rank uncertainty."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from accountpulse.contracts import capacity_metrics
from accountpulse.evidence import EVIDENCE, write_table

SEED = 20260922
CAPACITIES = [0.05, 0.10, 0.20, 0.30]
SCORES = {
    "G0_rfm": "rfm_score",
    "G0_logistic": "logistic_score",
    "G1_xgboost_cuda": "xgboost_score",
    "G2_cox_risk": "cox_risk",
    "G3_temporal_transformer_ensemble": "temporal_transformer_score",
    "G4_graphsage": "graphsage_score",
    "G5_text_embedding_mlp": "text_score",
    "G6_fixed_late_fusion": "g6_fixed_late_fusion",
    "P2_adaptive_fusion": "p2_adaptive_fusion",
    "G6_early_fusion_tabular_text": "g6_early_fusion_tabular_text",
}
PROBABILITIES = {
    "G0_logistic",
    "G1_xgboost_cuda",
    "G3_temporal_transformer_ensemble",
    "G4_graphsage",
    "G5_text_embedding_mlp",
    "G6_fixed_late_fusion",
    "P2_adaptive_fusion",
    "G6_early_fusion_tabular_text",
}


def calibration_stats(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
    top = probability >= np.quantile(probability, 0.9)
    return {
        "brier": brier_score_loss(y, probability),
        "log_loss": log_loss(y, clipped),
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0, 0]),
        "top_decile_predicted": float(probability[top].mean()),
        "top_decile_observed": float(y[top].mean()),
    }


def run(resamples: int = 5000) -> dict[str, Any]:
    predictions = pd.read_parquet(EVIDENCE / "uci_predictions.parquet")
    y = predictions["purchase_next_90d"].to_numpy()
    value = predictions["future_90d_spend"].to_numpy(float)
    curve_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for model, column in SCORES.items():
        score = predictions[column].to_numpy(float)
        for capacity in CAPACITIES:
            metric = capacity_metrics(score, value, y, capacity)
            curve_rows.append({"track": "C", "model": model, **metric.__dict__})
        if model in PROBABILITIES:
            calibration_rows.append({"track": "C", "model": model, **calibration_stats(y, score)})
    write_table("capacity_curves", pd.DataFrame(curve_rows))
    write_table("calibration_metrics", pd.DataFrame(calibration_rows))

    rng = np.random.default_rng(SEED)
    bootstrap_rows: list[dict[str, Any]] = []
    score_cache = {model: predictions[column].to_numpy(float) for model, column in SCORES.items()}
    for iteration in range(resamples):
        indices = rng.integers(0, len(predictions), len(predictions))
        for model in ("G0_rfm", "G1_xgboost_cuda", "G6_fixed_late_fusion", "P2_adaptive_fusion"):
            metric = capacity_metrics(score_cache[model][indices], value[indices], y[indices], 0.10)
            bootstrap_rows.append(
                {
                    "track": "C",
                    "iteration": iteration,
                    "model": model,
                    "cluster_unit": "customer_id",
                    "value_capture_10": metric.value_capture,
                    "win_capture_10": metric.win_capture,
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    write_table("bootstrap_results", bootstrap)
    pivot = bootstrap.pivot(index="iteration", columns="model", values="value_capture_10")
    difference = pivot["G6_fixed_late_fusion"] - pivot["G0_rfm"]
    comparison = {
        "estimate": float(
            capacity_metrics(score_cache["G6_fixed_late_fusion"], value, y, 0.10).value_capture
            - capacity_metrics(score_cache["G0_rfm"], value, y, 0.10).value_capture
        ),
        "ci_95": [float(difference.quantile(0.025)), float(difference.quantile(0.975))],
        "probability_positive": float((difference > 0).mean()),
        "resamples": resamples,
    }

    # Customer-cluster resampling rank membership, conditional on a customer being sampled.
    score = score_cache["G6_fixed_late_fusion"]
    selected_count = np.zeros(len(predictions), dtype=np.int32)
    present_count = np.zeros(len(predictions), dtype=np.int32)
    rank_samples: list[list[int]] = [[] for _ in range(len(predictions))]
    for _ in range(1000):
        indices = rng.integers(0, len(predictions), len(predictions))
        unique = np.unique(indices)
        present_count[unique] += 1
        order = unique[np.argsort(-score[unique], kind="stable")]
        cutoff = max(1, int(np.ceil(0.10 * len(unique))))
        selected_count[order[:cutoff]] += 1
        inverse_rank = np.empty(len(order), dtype=int)
        inverse_rank[np.arange(len(order))] = np.arange(1, len(order) + 1)
        for rank, row in enumerate(order, 1):
            rank_samples[int(row)].append(rank)
    p_top10 = selected_count / np.maximum(present_count, 1)
    uncertainty = predictions[["customer_id"]].copy()
    uncertainty["p_top10"] = p_top10
    uncertainty["median_rank"] = [
        float(np.median(values)) if values else np.nan for values in rank_samples
    ]
    uncertainty["review_flag"] = np.where(p_top10 >= 0.8, "AUTO_PRIORITIZE", "MANUAL_REVIEW")
    write_table("ranking_uncertainty", uncertainty)
    eligible = p_top10 >= 0.8
    selective = {
        "threshold": 0.8,
        "coverage": float(eligible.mean()),
        "value_capture": float(value[eligible].sum() / value.sum()) if value.sum() else 0.0,
        "win_capture": float(y[eligible].sum() / y.sum()) if y.sum() else 0.0,
    }

    snapshots = pd.read_parquet(EVIDENCE / "uci_lifecycle_snapshots.parquet")
    test = snapshots[snapshots["split"] == "TEST"].reset_index(drop=True)
    test["score"] = score_cache["G6_fixed_late_fusion"]
    test["value"] = value
    test["won"] = y
    test["cancellation_slice"] = np.where(
        test["cancellation_rate"] >= test["cancellation_rate"].quantile(0.9), "HIGH", "OTHER"
    )
    slice_rows: list[dict[str, Any]] = []
    for dimension in ("customer_cohort", "country", "cancellation_slice"):
        for group, subset in test.groupby(dimension):
            if len(subset) < 20 or subset["won"].nunique() < 2:
                continue
            metric = capacity_metrics(
                subset["score"].to_numpy(),
                subset["value"].to_numpy(),
                subset["won"].to_numpy(),
                0.10,
            )
            slice_rows.append(
                {
                    "track": "C",
                    "model": "G6_fixed_late_fusion",
                    "dimension": dimension,
                    "slice": str(group),
                    "n": len(subset),
                    "roc_auc": roc_auc_score(subset["won"], subset["score"]),
                    "value_capture_10": metric.value_capture,
                    "win_capture_10": metric.win_capture,
                }
            )
    write_table("robustness_slices", pd.DataFrame(slice_rows))
    report = {
        "status": "COMPLETE",
        "bootstrap_unit": "customer_id",
        "fusion_minus_rfm_value_capture_10": comparison,
        "selective_prioritization": selective,
        "note": "Bootstrap resamples stored predictions; models are not retrained.",
    }
    (EVIDENCE / "evaluation_result.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
