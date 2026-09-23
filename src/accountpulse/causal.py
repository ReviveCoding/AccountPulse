"""Randomized Criteo causal bridge, kept separate from predictive leaderboards."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlflow
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
import yaml
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from accountpulse.evidence import EVIDENCE, upsert_table, write_table
from accountpulse.gpu import GPUSlot, require_cuda
from accountpulse.paths import REPO
from accountpulse.provenance import validate_criteo_source

FEATURES = [f"f{i}" for i in range(12)]
OUTCOMES = ("visit", "conversion")
PROTOCOL_ID = "AP-V1-CAUSAL-BRIDGE-20260922-D2"


def splitmix64(values: np.ndarray, seed: int) -> NDArray[np.uint64]:
    """Stable 64-bit hash used for split assignment and outcome-blind tie resolution."""
    with np.errstate(over="ignore"):
        hashed = values.astype(np.uint64, copy=True) + np.uint64(seed) + np.uint64(
            0x9E3779B97F4A7C15
        )
        hashed = (hashed ^ (hashed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        hashed = (hashed ^ (hashed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return cast(NDArray[np.uint64], hashed ^ (hashed >> np.uint64(31)))


def hash_split(row_ids: np.ndarray, seed: int = 20260922) -> NDArray[np.str_]:
    """Return mutually exclusive 60/20/20 TRAIN/VALIDATION/TEST labels."""
    buckets = splitmix64(row_ids, seed) % np.uint64(100)
    return cast(
        NDArray[np.str_],
        np.where(buckets < 60, "TRAIN", np.where(buckets < 80, "VALIDATION", "TEST")),
    )


def deterministic_order(scores: np.ndarray, row_ids: np.ndarray) -> NDArray[np.intp]:
    """Descending score order with an outcome-independent deterministic tie breaker."""
    safe_scores = np.nan_to_num(scores, nan=-np.inf)
    tie_key = splitmix64(row_ids, 41_990_431)
    return cast(NDArray[np.intp], np.lexsort((tie_key, -safe_scores)))


def ipw_signal(
    treatment: np.ndarray, outcome: np.ndarray, propensity: float
) -> NDArray[np.float64]:
    """Horvitz-Thompson individual signal for a randomized binary treatment."""
    if not 0 < propensity < 1:
        raise ValueError("propensity must be in (0, 1)")
    return cast(
        NDArray[np.float64],
        treatment * outcome / propensity - (1 - treatment) * outcome / (1 - propensity),
    )


def aipw_signal(
    treatment: np.ndarray,
    outcome: np.ndarray,
    propensity: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    clip: tuple[float, float] = (0.05, 0.95),
) -> NDArray[np.float64]:
    """Doubly robust conditional-effect signal for held-out randomized evidence."""
    bounded = np.clip(propensity, clip[0], clip[1])
    return cast(
        NDArray[np.float64],
        mu1
        - mu0
        + treatment * (outcome - mu1) / bounded
        - (1 - treatment) * (outcome - mu0) / (1 - bounded),
    )


def _policy_metrics_from_signal(
    scores: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    row_ids: np.ndarray,
    capacities: list[float],
    signal: np.ndarray,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    order = deterministic_order(scores, row_ids)
    cumulative: NDArray[np.float64] = np.cumsum(signal[order], dtype=np.float64)
    n = len(order)
    rows: list[dict[str, Any]] = []
    for capacity in capacities:
        selected_n = max(1, int(np.ceil(capacity * n)))
        selected = order[:selected_n]
        treated = treatment[selected] == 1
        control = ~treated
        treated_mean = float(outcome[selected][treated].mean())
        control_mean = float(outcome[selected][control].mean())
        rows.append(
            {
                "capacity": capacity,
                "selected_n": float(selected_n),
                "treatment_rate": float(treated.mean()),
                "treated_outcome_rate": treated_mean,
                "control_outcome_rate": control_mean,
                "incremental_rate_selected_unadjusted": treated_mean - control_mean,
                "aipw_incremental_per_eligible": float(cumulative[selected_n - 1] / n),
            }
        )
    fractions = np.linspace(0.0, 1.0, 101)
    gains = np.zeros_like(fractions)
    positive = fractions > 0
    positions = np.ceil(fractions[positive] * n).astype(np.int64) - 1
    gains[positive] = cumulative[positions] / n
    overall_ate = float(cumulative[-1] / n)
    qini = gains - fractions * overall_ate
    curve = pd.DataFrame(
        {"capacity": fractions, "policy_gain": gains, "qini_gain": qini}
    )
    curve.attrs["auuc"] = float(np.trapezoid(gains, fractions))
    curve.attrs["qini_coefficient"] = float(np.trapezoid(qini, fractions))
    return rows, curve


def policy_metrics(
    scores: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    row_ids: np.ndarray,
    capacities: list[float],
    propensity: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Compute randomized policy values plus AUUC/Qini curve from fixed scores."""
    signal = ipw_signal(treatment, outcome, propensity)
    rows, curve = _policy_metrics_from_signal(
        scores, treatment, outcome, row_ids, capacities, signal
    )
    for row in rows:
        row["ipw_incremental_per_eligible"] = row.pop("aipw_incremental_per_eligible")
    return rows, curve


def _bootstrap_ci(
    contribution: np.ndarray,
    row_ids: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    """Memory-bounded deterministic bucket bootstrap over stored predictions."""
    buckets = (splitmix64(row_ids, seed) % np.uint64(512)).astype(np.int16)
    sums = np.bincount(buckets, weights=contribution, minlength=512)
    counts = np.bincount(buckets, minlength=512)
    rng = np.random.default_rng(seed)
    estimates: NDArray[np.float64] = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 250):
        width = min(250, replicates - start)
        weights = rng.poisson(1.0, size=(width, 512))
        estimates[start : start + width] = (weights @ sums) / (weights @ counts)
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
        float((estimates > 0).mean()),
    )


def _fit_booster(
    features: np.ndarray,
    target: np.ndarray,
    config: dict[str, Any],
    *,
    regression: bool = False,
) -> xgb.Booster:
    matrix = xgb.QuantileDMatrix(features, target, max_bin=config["max_bin"])
    params: dict[str, Any] = {
        "objective": "reg:squarederror" if regression else "binary:logistic",
        "eval_metric": "rmse" if regression else "logloss",
        "device": "cuda",
        "tree_method": "hist",
        "max_depth": config["max_depth"],
        "eta": config["eta"],
        "max_bin": config["max_bin"],
        "subsample": config["subsample"],
        "sampling_method": "gradient_based",
        "seed": config["seed"],
        "nthread": 4,
    }
    return xgb.train(params, matrix, num_boost_round=config["num_boost_round"])


def _predict(model: xgb.Booster, features: np.ndarray) -> NDArray[np.float32]:
    return cast(
        NDArray[np.float32], np.asarray(model.inplace_predict(features), dtype=np.float32)
    )


def _save_model(model: xgb.Booster, name: str) -> tuple[str, str]:
    directory = REPO / "artifacts/models/criteo"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    model.save_model(path)
    return str(path.relative_to(REPO)), hashlib.sha256(path.read_bytes()).hexdigest()


class _GpuMemorySampler:
    def __init__(self) -> None:
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(0.2):
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                values = [int(line.strip()) for line in output.splitlines() if line.strip()]
                self.peak_mib = max([self.peak_mib, *values])
            except (OSError, ValueError, subprocess.SubprocessError):
                pass

    def __enter__(self) -> _GpuMemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()


@dataclass
class DataSplit:
    row_ids: np.ndarray
    features: np.ndarray
    treatment: np.ndarray
    visit: np.ndarray
    conversion: np.ndarray


def _load_data(path: Path, split_seed: int) -> dict[str, DataSplit]:
    validate_criteo_source(path)
    schema = {name: pl.Float32 for name in FEATURES} | {
        "treatment": pl.UInt8,
        "visit": pl.UInt8,
        "conversion": pl.UInt8,
    }
    frame = pl.read_csv(
        path,
        columns=[*FEATURES, "treatment", "visit", "conversion"],
        schema_overrides=schema,
        low_memory=True,
    ).rechunk()
    row_ids = np.arange(frame.height, dtype=np.uint64)
    buckets = splitmix64(row_ids, split_seed) % np.uint64(100)
    all_features = frame.select(FEATURES).to_numpy()
    all_treatment = frame["treatment"].to_numpy()
    all_visit = frame["visit"].to_numpy()
    all_conversion = frame["conversion"].to_numpy()
    result: dict[str, DataSplit] = {}
    masks = {
        "TRAIN": buckets < 60,
        "VALIDATION": (buckets >= 60) & (buckets < 80),
        "TEST": buckets >= 80,
    }
    for label, mask in masks.items():
        result[label] = DataSplit(
            row_ids=row_ids[mask],
            features=all_features[mask],
            treatment=all_treatment[mask],
            visit=all_visit[mask],
            conversion=all_conversion[mask],
        )
    return result


def _concatenate_splits(left: DataSplit, right: DataSplit) -> DataSplit:
    return DataSplit(
        row_ids=np.concatenate((left.row_ids, right.row_ids)),
        features=np.concatenate((left.features, right.features)),
        treatment=np.concatenate((left.treatment, right.treatment)),
        visit=np.concatenate((left.visit, right.visit)),
        conversion=np.concatenate((left.conversion, right.conversion)),
    )


def _uplift_scores(
    train: DataSplit,
    evaluation: DataSplit,
    outcome: str,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, str]]]:
    y = getattr(train, outcome)
    treated = train.treatment == 1
    treatment_model = _fit_booster(train.features[treated], y[treated], config)
    control_model = _fit_booster(train.features[~treated], y[~treated], config)
    mu1 = _predict(treatment_model, evaluation.features)
    mu0 = _predict(control_model, evaluation.features)

    s_features = np.column_stack((train.features, train.treatment.astype(np.float32)))
    s_model = _fit_booster(s_features, y, config)
    del s_features
    s0 = _predict(
        s_model, np.column_stack((evaluation.features, np.zeros(len(evaluation.features))))
    )
    s1 = _predict(
        s_model, np.column_stack((evaluation.features, np.ones(len(evaluation.features))))
    )

    propensity = 0.85
    transformed: NDArray[np.float32] = ipw_signal(
        train.treatment, y, propensity
    ).astype(np.float32)
    transformed_model = _fit_booster(train.features, transformed, config, regression=True)
    scores = {
        "response": mu1,
        "s_learner": s1 - s0,
        "t_learner": mu1 - mu0,
        "transformed_outcome": _predict(transformed_model, evaluation.features),
    }
    models = {
        f"{outcome}_treatment_response": _save_model(
            treatment_model, f"{outcome}_treatment_response"
        ),
        f"{outcome}_t_control": _save_model(control_model, f"{outcome}_t_control"),
        f"{outcome}_s_learner": _save_model(s_model, f"{outcome}_s_learner"),
        f"{outcome}_transformed": _save_model(transformed_model, f"{outcome}_transformed"),
    }
    return scores, models


def _selected_outcome_scores(
    train: DataSplit,
    evaluation: DataSplit,
    outcome: str,
    selected_method: str,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, str]], dict[str, np.ndarray]]:
    all_scores, models = _uplift_scores(train, evaluation, outcome, config)
    rankings = {
        "response": all_scores["response"],
        "selected_uplift": all_scores[selected_method],
    }
    nuisance = {
        "mu1": all_scores["response"],
        "mu0": all_scores["response"] - all_scores["t_learner"],
    }
    return rankings, models, nuisance


def _balance_rows(splits: dict[str, DataSplit]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, data in splits.items():
        treated = data.treatment == 1
        asmd_values = []
        for column, feature in enumerate(FEATURES):
            left = data.features[treated, column]
            right = data.features[~treated, column]
            denominator = np.sqrt((left.var() + right.var()) / 2)
            asmd = float(abs(left.mean() - right.mean()) / denominator) if denominator else 0.0
            asmd_values.append(asmd)
            rows.append(
                {
                    "protocol_id": PROTOCOL_ID,
                    "split": name,
                    "feature": feature,
                    "asmd": asmd,
                    "treatment_rate": float(treated.mean()),
                    "sample_count": len(data.treatment),
                }
            )
        rows.append(
            {
                "protocol_id": PROTOCOL_ID,
                "split": name,
                "feature": "MAX_ABSOLUTE_SMD",
                "asmd": max(asmd_values),
                "treatment_rate": float(treated.mean()),
                "sample_count": len(data.treatment),
            }
        )
    return pd.DataFrame(rows)


def run() -> dict[str, Any]:
    """Execute E28 without reading or changing Track-B/C evidence."""
    started = time.perf_counter()
    config = yaml.safe_load((REPO / "configs/causal.yaml").read_text(encoding="utf-8"))
    if config["protocol_id"] != PROTOCOL_ID:
        raise ValueError("Causal config/code protocol identity mismatch")
    source = REPO / "data/incoming/criteo/criteo-research-uplift-v2.1.csv.gz"
    observed_hash = validate_criteo_source(source)
    splits = _load_data(source, config["split_seed"])
    split_rows = [
        {
            "protocol_id": PROTOCOL_ID,
            "split": name,
            "sample_count": len(data.row_ids),
            "treatment_count": int(data.treatment.sum()),
            "control_count": int((1 - data.treatment).sum()),
            "visit_count": int(data.visit.sum()),
            "conversion_count": int(data.conversion.sum()),
        }
        for name, data in splits.items()
    ]
    write_table("causal_split_summary", pd.DataFrame(split_rows))
    write_table("causal_balance", _balance_rows(splits))

    model_files: dict[str, tuple[str, str]] = {}
    fit_started = time.perf_counter()
    with GPUSlot(timeout=5), _GpuMemorySampler() as gpu_sampler:
        require_cuda()
        evaluation = _concatenate_splits(splits["VALIDATION"], splits["TEST"])
        combined_propensity_model = _fit_booster(
            splits["TRAIN"].features,
            splits["TRAIN"].treatment,
            config["xgboost"],
        )
        combined_propensity = _predict(combined_propensity_model, evaluation.features)
        model_files["treatment_propensity_nuisance"] = _save_model(
            combined_propensity_model, "treatment_propensity_nuisance"
        )
        combined_visit_scores, models = _uplift_scores(
            splits["TRAIN"], evaluation, "visit", config["xgboost"]
        )
        model_files.update(models)
        validation_n = len(splits["VALIDATION"].row_ids)
        validation_scores = {
            method: scores[:validation_n] for method, scores in combined_visit_scores.items()
        }
        validation_propensity = combined_propensity[:validation_n]
        validation_signal = aipw_signal(
            splits["VALIDATION"].treatment,
            splits["VALIDATION"].visit,
            validation_propensity,
            validation_scores["response"] - validation_scores["t_learner"],
            validation_scores["response"],
            tuple(config["propensity_clip"]),
        )
        selection_rows = []
        for method in ("s_learner", "t_learner", "transformed_outcome"):
            metrics, curve = _policy_metrics_from_signal(
                validation_scores[method],
                splits["VALIDATION"].treatment,
                splits["VALIDATION"].visit,
                splits["VALIDATION"].row_ids,
                [0.10],
                validation_signal,
            )
            selection_rows.append(
                {
                    "protocol_id": PROTOCOL_ID,
                    "method": method,
                    "validation_policy_value_at_10": metrics[0][
                        "aipw_incremental_per_eligible"
                    ],
                    "validation_auuc": curve.attrs["auuc"],
                    "validation_qini": curve.attrs["qini_coefficient"],
                }
            )
        selection = pd.DataFrame(selection_rows).sort_values(
            ["validation_policy_value_at_10", "method"], ascending=[False, True]
        )
        selected_method = str(selection.iloc[0]["method"])
        selection["selected_before_test"] = selection["method"] == selected_method
        write_table("causal_model_selection", selection)

        test_visit_scores = {
            method: scores[validation_n:] for method, scores in combined_visit_scores.items()
        }
        test_propensity = combined_propensity[validation_n:]
        test_conversion_scores, models, conversion_nuisance = _selected_outcome_scores(
            splits["TRAIN"],
            splits["TEST"],
            "conversion",
            selected_method,
            config["xgboost"],
        )
        model_files.update(models)
        peak_vram_mib = gpu_sampler.peak_mib or None
    training_seconds = time.perf_counter() - fit_started

    test = splits["TEST"]
    prediction_data: dict[str, Any] = {
        "protocol_id": pd.Categorical([PROTOCOL_ID] * len(test.row_ids)),
        "row_id": test.row_ids,
        "treatment": test.treatment,
        "visit": test.visit,
        "conversion": test.conversion,
        "propensity_nuisance": test_propensity,
    }
    for method, score in test_visit_scores.items():
        prediction_data[f"visit_{method}"] = score
    for method, score in test_conversion_scores.items():
        prediction_data[f"conversion_{method}"] = score
    predictions = pd.DataFrame(prediction_data)
    write_table("causal_predictions", predictions)

    policy_rows: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, Any]] = []
    primary_selected = test_visit_scores[selected_method]
    outcome_scores = {
        "visit": test_visit_scores,
        "conversion": test_conversion_scores,
    }
    nuisance_by_outcome = {
        "visit": {
            "mu1": test_visit_scores["response"],
            "mu0": test_visit_scores["response"] - test_visit_scores["t_learner"],
        },
        "conversion": conversion_nuisance,
    }
    for outcome, scores_by_method in outcome_scores.items():
        y = getattr(test, outcome)
        nuisance = nuisance_by_outcome[outcome]
        signal = aipw_signal(
            test.treatment,
            y,
            test_propensity,
            nuisance["mu0"],
            nuisance["mu1"],
            tuple(config["propensity_clip"]),
        )
        for method, scores in scores_by_method.items():
            metrics, curve = _policy_metrics_from_signal(
                scores,
                test.treatment,
                y,
                test.row_ids,
                config["capacities"],
                signal,
            )
            for metric in metrics:
                selected_n = int(metric["selected_n"])
                selected = deterministic_order(scores, test.row_ids)[:selected_n]
                contribution: NDArray[np.float64] = np.zeros(
                    len(test.row_ids), dtype=np.float64
                )
                contribution[selected] = signal[selected]
                low, high, probability_positive = _bootstrap_ci(
                    contribution,
                    test.row_ids,
                    config["bootstrap_replicates"],
                    config["split_seed"] + selected_n,
                )
                metric.update(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "outcome": outcome,
                        "method": method,
                        "ci_low": low,
                        "ci_high": high,
                        "probability_positive": probability_positive,
                    }
                )
                policy_rows.append(metric)
                bootstrap_rows.append(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "outcome": outcome,
                        "method": method,
                        "capacity": metric["capacity"],
                        "estimate": metric["aipw_incremental_per_eligible"],
                        "ci_low": low,
                        "ci_high": high,
                        "replicates": config["bootstrap_replicates"],
                        "unit": "deterministic_row_hash_bucket",
                    }
                )
            curve.insert(0, "method", method)
            curve.insert(0, "outcome", outcome)
            curve.insert(0, "protocol_id", PROTOCOL_ID)
            curve["auuc"] = curve.attrs["auuc"]
            curve["qini_coefficient"] = curve.attrs["qini_coefficient"]
            curve_frames.append(curve)

    comparison_rows: list[dict[str, Any]] = []
    for capacity in config["capacities"]:
        n_selected = max(1, int(np.ceil(capacity * len(test.row_ids))))
        response_selected = deterministic_order(
            test_visit_scores["response"], test.row_ids
        )[:n_selected]
        uplift_selected = deterministic_order(primary_selected, test.row_ids)[:n_selected]
        response_flag: NDArray[np.bool_] = np.zeros(len(test.row_ids), dtype=bool)
        uplift_flag: NDArray[np.bool_] = np.zeros(len(test.row_ids), dtype=bool)
        response_flag[response_selected] = True
        uplift_flag[uplift_selected] = True
        signal = aipw_signal(
            test.treatment,
            test.visit,
            test_propensity,
            test_visit_scores["response"] - test_visit_scores["t_learner"],
            test_visit_scores["response"],
            tuple(config["propensity_clip"]),
        )
        delta = (uplift_flag.astype(float) - response_flag.astype(float)) * signal
        low, high, probability_positive = _bootstrap_ci(
            delta,
            test.row_ids,
            config["bootstrap_replicates"],
            config["split_seed"] + n_selected + 71,
        )
        intersection = int((response_flag & uplift_flag).sum())
        union = int((response_flag | uplift_flag).sum())
        comparison_rows.append(
            {
                "protocol_id": PROTOCOL_ID,
                "outcome": "visit",
                "capacity": capacity,
                "response_method": "response",
                "uplift_method": selected_method,
                "overlap_fraction_of_each_list": intersection / n_selected,
                "jaccard": intersection / union,
                "uplift_minus_response_aipw_value": float(delta.mean()),
                "ci_low": low,
                "ci_high": high,
                "probability_positive": probability_positive,
                "bootstrap_replicates": config["bootstrap_replicates"],
            }
        )

    write_table("causal_policy_metrics", pd.DataFrame(policy_rows))
    write_table("causal_uplift_curves", pd.concat(curve_frames, ignore_index=True))
    write_table("causal_bootstrap_results", pd.DataFrame(bootstrap_rows))
    write_table("causal_policy_comparison", pd.DataFrame(comparison_rows))

    systems = pd.DataFrame(
        [
            {
                "protocol_id": PROTOCOL_ID,
                "stage": "E28_CAUSAL_BRIDGE",
                "gpu_identity": "NVIDIA GeForce RTX 4090 Laptop GPU",
                "precision": "float32_hist",
                "training_wall_seconds": training_seconds,
                "total_wall_seconds": time.perf_counter() - started,
                "peak_gpu_memory_mib_nvidia_smi": peak_vram_mib,
                "sample_count": sum(len(item.row_ids) for item in splits.values()),
                "train_count": len(splits["TRAIN"].row_ids),
                "validation_count": len(splits["VALIDATION"].row_ids),
                "test_count": len(splits["TEST"].row_ids),
                "model_identity": (
                    "XGBoost CUDA propensity/response/S/T/transformed-outcome; "
                    "host-array inference"
                ),
            }
        ]
    )
    upsert_table("systems_metrics", systems, ["protocol_id", "stage"])

    comparison = pd.DataFrame(comparison_rows)
    result = {
        "protocol_id": PROTOCOL_ID,
        "source_sha256": observed_hash,
        "source_revision": config["source_revision"],
        "split_design": "stable row-index hash 60/20/20; no business time is available",
        "split_counts": {row["split"]: row["sample_count"] for row in split_rows},
        "selected_uplift_method_from_validation": selected_method,
        "estimator": (
            "AIPW with train-only XGBoost propensity and outcome nuisance models; "
            "propensity clipped to [0.05, 0.95]"
        ),
        "propensity_diagnostics": {
            "validation_auc": float(
                roc_auc_score(splits["VALIDATION"].treatment, validation_propensity)
            ),
            "test_auc": float(roc_auc_score(test.treatment, test_propensity)),
            "validation_min": float(validation_propensity.min()),
            "validation_max": float(validation_propensity.max()),
            "test_min": float(test_propensity.min()),
            "test_max": float(test_propensity.max()),
            "test_clip_fraction": float(
                ((test_propensity < config["propensity_clip"][0])
                | (test_propensity > config["propensity_clip"][1])).mean()
            ),
        },
        "primary_outcome": "visit",
        "secondary_outcome": "conversion",
        "treatment_predictor_policy": (
            "treatment excluded from response/T-learner feature vectors; S-learner uses it "
            "only as the randomized intervention; exposure excluded"
        ),
        "primary_comparison": comparison.to_dict(orient="records"),
        "training_wall_seconds": training_seconds,
        "peak_gpu_memory_mib_nvidia_smi": peak_vram_mib,
        "model_artifacts": {
            name: {"path": path, "sha256": digest}
            for name, (path, digest) in model_files.items()
        },
        "claim_boundary": (
            "Randomized public advertising benchmark only; no transportability to "
            "Maven, Olist, UCI, enterprise CRM, or production outreach."
        ),
    }
    result_path = EVIDENCE / "causal_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    mlflow.set_tracking_uri(f"sqlite:///{EVIDENCE / 'mlflow.db'}")
    mlflow.set_experiment("accountpulse-causal-bridge")
    with mlflow.start_run(run_name=PROTOCOL_ID):
        mlflow.log_params(
            {
                "protocol_id": PROTOCOL_ID,
                "source_sha256": observed_hash,
                "source_revision": config["source_revision"],
                "split": "hash_60_20_20",
                "selected_uplift_method": selected_method,
                "test_used_for_selection": False,
            }
        )
        at_ten = comparison.loc[comparison["capacity"] == 0.10].iloc[0]
        mlflow.log_metrics(
            {
                "visit_uplift_minus_response_value_at_10": at_ten[
                    "uplift_minus_response_aipw_value"
                ],
                "visit_top10_overlap": at_ten["overlap_fraction_of_each_list"],
                "training_wall_seconds": training_seconds,
            }
        )
        mlflow.log_artifact(str(result_path))
        mlflow.log_artifact(str(REPO / "configs/causal.yaml"))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
