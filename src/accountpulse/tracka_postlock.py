"""Post-lock Track-A diagnostics that cannot alter the frozen decision."""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from accountpulse.evidence import EVIDENCE, upsert_table, write_table
from accountpulse.tracka_data import PROTOCOL
from accountpulse.tracka_models import BUNDLE_PATH, CAPACITIES, FrozenTrackAScorer


def _rank_uncertainty(frame: pd.DataFrame, replicates: int = 10_000) -> pd.DataFrame:
    """Cluster-bootstrap conditional ranks using immutable stored predictions."""
    keys = frame["account_key"].astype(str).to_numpy()
    groups = np.unique(keys)
    indices = {key: np.flatnonzero(keys == key) for key in groups}
    score = frame["AccountPulse_EV"].to_numpy(float)
    rng = np.random.default_rng(20260923)
    ranks: list[list[int]] = [[] for _ in range(len(frame))]
    selected = {0.05: np.zeros(len(frame)), 0.10: np.zeros(len(frame)), 0.20: np.zeros(len(frame))}
    appearances = np.zeros(len(frame))
    for _ in range(replicates):
        sampled = rng.choice(groups, len(groups), replace=True)
        take = np.concatenate([indices[key] for key in sampled])
        order = take[np.argsort(-score[take], kind="stable")]
        for rank, index in enumerate(order, start=1):
            ranks[index].append(rank)
            appearances[index] += 1
        for capacity, counts in selected.items():
            top = order[: max(1, int(np.ceil(capacity * len(order))))]
            np.add.at(counts, top, 1)
    rows = []
    for index, values in enumerate(ranks):
        rows.append(
            {
                "protocol_id": PROTOCOL,
                "opportunity_id": frame.iloc[index]["opportunity_id"],
                "median_rank": float(np.median(values)),
                "rank_q025": float(np.quantile(values, 0.025)),
                "rank_q975": float(np.quantile(values, 0.975)),
                "p_top5": float(selected[0.05][index] / max(appearances[index], 1)),
                "p_top10": float(selected[0.10][index] / max(appearances[index], 1)),
                "p_top20": float(selected[0.20][index] / max(appearances[index], 1)),
                "replicates": replicates,
                "method": "account_cluster_bootstrap_stored_predictions",
            }
        )
    return pd.DataFrame(rows)


def run() -> dict[str, Any]:
    result = json.loads((EVIDENCE / "maven_locked_result.json").read_text(encoding="utf-8"))
    predictions = pd.read_parquet(EVIDENCE / "maven_locked_predictions.parquet")
    uncertainty = _rank_uncertainty(predictions)
    write_table("maven_ranking_uncertainty_locked", uncertainty)

    model_columns = [
        name
        for name in predictions.columns
        if name.startswith(("B0_", "B1_", "B2_", "B3_", "B4_", "D0_", "D1_", "D2_", "D3_", "D4_"))
        or name in {"LambdaMART", "AccountPulse_EV", "POSTASSIGN_xgboost"}
    ]
    secondary = []
    for model in model_columns:
        order = np.argsort(-predictions[model].to_numpy(float), kind="stable")
        for capacity in CAPACITIES:
            count = max(1, int(np.ceil(capacity * len(predictions))))
            top = order[:count]
            wins = predictions["won"].to_numpy(int)
            values = predictions["realized_value"].to_numpy(float)
            gains = values[order]
            discount = 1 / np.log2(np.arange(2, len(gains) + 2))
            ideal = np.sort(values)[::-1]
            denominator = float((ideal * discount).sum())
            secondary.append(
                {
                    "protocol_id": PROTOCOL,
                    "model": model,
                    "capacity": capacity,
                    "precision_at_k": float(wins[top].mean()),
                    "recall_at_k": float(wins[top].sum() / max(wins.sum(), 1)),
                    "ndcg": float((gains * discount).sum() / max(denominator, 1e-12)),
                }
            )
    write_table("maven_ranking_secondary_locked", pd.DataFrame(secondary))

    scorer: FrozenTrackAScorer = joblib.load(BUNDLE_PATH)
    features = pd.read_parquet(EVIDENCE / "maven_locked_features.parquet")
    encoded = scorer._encoded(features)
    timing = scorer.timing_model.predict_proba(encoded)
    survival_rows = []
    briers = []
    for horizon, probability in ((30, timing[:, 1]), (60, timing[:, 1] + timing[:, 2])):
        event = (
            predictions["event_within_horizon"].to_numpy(bool)
            & (predictions["event_time"].to_numpy(float) <= horizon)
        ).astype(int)
        brier = float(brier_score_loss(event, probability))
        briers.append(brier)
        survival_rows.extend(
            [
                {
                    "protocol_id": PROTOCOL,
                    "model": "B8_discrete_terminal",
                    "metric": f"AUC@{horizon}",
                    "value": float(roc_auc_score(event, probability)),
                },
                {
                    "protocol_id": PROTOCOL,
                    "model": "B8_discrete_terminal",
                    "metric": f"Brier@{horizon}",
                    "value": brier,
                },
            ]
        )
    survival_rows.append(
        {
            "protocol_id": PROTOCOL,
            "model": "B8_discrete_terminal",
            "metric": "discrete_integrated_brier_mean_30_60",
            "value": float(np.mean(briers)),
        }
    )
    competing_target = np.select(
        [predictions["won"].eq(1), predictions["lost"].eq(1)], [1, 2], default=0
    )
    competing_probability = scorer.competing_model.predict_proba(encoded)
    survival_rows.append(
        {
            "protocol_id": PROTOCOL,
            "model": "B8_competing_terminal",
            "metric": "multiclass_log_loss",
            "value": float(log_loss(competing_target, competing_probability, labels=[0, 1, 2])),
        }
    )
    write_table("maven_survival_metrics_locked", pd.DataFrame(survival_rows))
    value_rows = []
    actual = predictions["realized_value"].to_numpy(float)
    for model, estimate in (
        ("B11_xgboost_direct", predictions["D1_predicted_realized_value"].to_numpy(float)),
        ("D2_probability_times_value", predictions["D2_probability_times_value"].to_numpy(float)),
    ):
        value_rows.append(
            {
                "protocol_id": PROTOCOL,
                "model": model,
                "mae": float(mean_absolute_error(actual, estimate)),
                "rmse": float(mean_squared_error(actual, estimate) ** 0.5),
            }
        )
    write_table("maven_value_metrics_locked", pd.DataFrame(value_rows))

    development = pd.read_parquet(EVIDENCE / "maven_development.parquet")
    policy = development[development["split"] == "POLICY"]
    monitoring = []
    for field in ("revenue", "employees", "sales_price", "account_prior_opportunities"):
        monitoring.extend(
            [
                {
                    "protocol_id": PROTOCOL,
                    "metric_type": "input_drift_ks",
                    "field": field,
                    "value": float(
                        ks_2samp(policy[field].dropna(), features[field].dropna()).statistic
                    ),
                    "reference": "POLICY",
                    "current": "LOCKED",
                },
                {
                    "protocol_id": PROTOCOL,
                    "metric_type": "missingness",
                    "field": field,
                    "value": float(features[field].isna().mean()),
                    "reference": "POLICY",
                    "current": "LOCKED",
                },
            ]
        )
    monitoring.append(
        {
            "protocol_id": PROTOCOL,
            "metric_type": "score_drift_ks",
            "field": "AccountPulse_EV",
            "value": float(
                ks_2samp(
                    pd.read_parquet(EVIDENCE / "maven_predictions_development.parquet").loc[
                        lambda x: x["split"] == "POLICY", "AccountPulse_EV"
                    ],
                    predictions["AccountPulse_EV"],
                ).statistic
            ),
            "reference": "POLICY",
            "current": "LOCKED",
        }
    )
    write_table("maven_monitoring_metrics", pd.DataFrame(monitoring))
    shadow = pd.DataFrame(
        [
            {
                "step": 1,
                "state": "SCORE",
                "status": "COMPLETE",
                "detail": "frozen A1 LOCKED batch scored",
            },
            {
                "step": 2,
                "state": "WAIT_FOR_MATURITY",
                "status": "COMPLETE",
                "detail": "60-day outcome window observed",
            },
            {
                "step": 3,
                "state": "EVALUATE",
                "status": "COMPLETE",
                "detail": "10,000 account-cluster resamples",
            },
            {
                "step": 4,
                "state": "DETECT_DRIFT",
                "status": "COMPLETE",
                "detail": "PIT input and score KS diagnostics",
            },
            {
                "step": 5,
                "state": "TRAIN_CHALLENGER",
                "status": "COMPLETE_PRE_LOCK",
                "detail": "AP-EV frozen before outcome exposure",
            },
            {
                "step": 6,
                "state": "VALIDATE",
                "status": "COMPLETE_PRE_LOCK",
                "detail": "VALIDATION/POLICY only",
            },
            {
                "step": 7,
                "state": "SHADOW",
                "status": "COMPLETE",
                "detail": "stored one-shot LOCKED predictions",
            },
            {
                "step": 8,
                "state": "PROMOTE_OR_REJECT",
                "status": "RETAIN_BASELINE",
                "detail": "G5 and G7 failed; G9 not evaluable",
            },
        ]
    ).assign(protocol_id=PROTOCOL)
    write_table("maven_shadow_replay", shadow)

    ranking = pd.concat(
        [
            pd.read_parquet(EVIDENCE / "maven_ranking_metrics_development.parquet"),
            pd.read_parquet(EVIDENCE / "maven_ranking_metrics_locked.parquet"),
        ],
        ignore_index=True,
    )
    upsert_table("ranking_metrics", ranking, ["protocol_id", "cohort", "model", "capacity"])
    calibration = pd.read_parquet(EVIDENCE / "maven_calibration_metrics_locked.parquet").assign(
        protocol_id=PROTOCOL, track="A", cohort="LOCKED"
    )
    upsert_table("calibration_metrics", calibration, ["protocol_id", "cohort", "method"])
    robustness = pd.read_parquet(EVIDENCE / "maven_robustness_slices_locked.parquet")
    upsert_table(
        "robustness_slices",
        robustness,
        ["protocol_id", "slice", "slice_value", "model"],
    )
    bootstrap = pd.DataFrame(
        [
            {
                "protocol_id": PROTOCOL,
                "track": "A",
                "cohort": "LOCKED",
                "comparison": "AccountPulse_EV-minus-" + result["strongest_frozen_baseline"],
                "metric": metric,
                "estimate": values["estimate"],
                "ci_low": values["ci_95"][0],
                "ci_high": values["ci_95"][1],
                "replicates": result["bootstrap"]["replicates"],
                "cluster": result["bootstrap"]["cluster"],
            }
            for metric, values in (
                ("value_capture_difference", result["bootstrap"]["value_capture_difference"]),
                ("win_capture_difference", result["bootstrap"]["win_capture_difference"]),
            )
        ]
    )
    upsert_table("bootstrap_results", bootstrap, ["protocol_id", "cohort", "metric"])
    return {
        "protocol_id": PROTOCOL,
        "status": "COMPLETE_POST_LOCK_DIAGNOSTICS",
        "ranking_uncertainty_rows": len(uncertainty),
        "ranking_secondary_rows": len(secondary),
        "survival_metric_rows": len(survival_rows),
        "value_metric_rows": len(value_rows),
        "monitoring_metric_rows": len(monitoring),
        "shadow_replay": "COMPLETE_HISTORICAL_REPLAY_NOT_ONLINE_DEPLOYMENT",
        "decision_unchanged": result["decision"],
    }
