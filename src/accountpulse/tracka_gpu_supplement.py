# ruff: noqa: E501, RUF001
"""GPU infrastructure-recovery supplement for frozen Track-A diagnostics.

Only read-only inference from the A1 frozen bundle is permitted.  The original
DIAG1 infrastructure-failure record is preserved as historical evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import yaml
from scipy.special import expit

from accountpulse.evidence import EVIDENCE, write_table
from accountpulse.paths import REPO
from accountpulse.tracka_data import PREASSIGN_CATEGORICAL, PREASSIGN_NUMERIC
from accountpulse.tracka_models import BUNDLE_PATH, FrozenTrackAScorer, _calibrate

BASE_PROTOCOL = "AP-V1-TRACKA-20260923-A1"
DIAGNOSTIC_PROTOCOL = "AP-V1-TRACKA-20260923-A1-DIAG1"
SUPPLEMENT_PROTOCOL = "AP-V1-TRACKA-20260923-A1-DIAG1-GPU1"
LABEL = "POST_LOCK_DIAGNOSTIC_ONLY"
BASE_COMMIT = "4d7e641a054790bb938b61f39d68cfd7b76aeb4b"
EXPECTED_PROTECTED = {
    "FREEZE_MANIFEST.yaml": "7a889b633b40b0dd73c2d392d773c409316250e7456da5089f9f4b1542ad5a33",
    "LOCKED_START_RECEIPT.json": "3cbff875cc282741f599893c7a4a4a55e2158003b2566c68e88de59f8eba1562",
    "artifacts/evidence/maven_locked_result.json": "9f0283aa77c3b18cab6853fac43490eb14732e0ba2c20a124587e9deea41aba1",
    "configs/track_a.yaml": "4a717642a36d45147dabdf2159d2807df24c23363b26a884ef292f6970505113",
    "artifacts/models/tracka/tracka_frozen_bundle.joblib": "e5328b5a588a7e9abab60dd898a8782b4af67cd69790ff61801f8dd9f423caba",
}
OUTPUT_NAMES = (
    "maven_b4_shap_global_gpu_postlock",
    "maven_b4_shap_attribution_drift_gpu_postlock",
    "maven_b4_shap_local_gpu_postlock",
    "maven_apev_components_gpu_postlock",
    "maven_grouped_permutation_gpu_postlock",
    "maven_b4_pdp_ice_gpu_postlock",
)
FEATURE_GROUPS = (
    "firmographics",
    "product",
    "calendar",
    "account_history",
    "product_history",
    "sector_history",
)
PDP_FEATURES = (
    "revenue",
    "sales_price",
    "employees",
    "account_prior_win_rate",
    "account_prior_opportunities",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes() -> dict[str, str]:
    """Hash every immutable A1 input used by the supplement."""
    return {name: _sha256(REPO / name) for name in EXPECTED_PROTECTED}


def _meta(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    metadata = {
        "base_protocol": BASE_PROTOCOL,
        "diagnostic_protocol": DIAGNOSTIC_PROTOCOL,
        "supplement_protocol": SUPPLEMENT_PROTOCOL,
        "result_label": LABEL,
        "post_lock": True,
        "decision_eligible": False,
        "infrastructure_recovery": True,
        "a1_decision": "RETAIN_BASELINE",
    }
    for position, (key, value) in enumerate(metadata.items()):
        result.insert(position, key, value)
    return result


def verify_immutability_gate() -> dict[str, str]:
    """Fail before inference unless branch, base, hashes, and DIAG1 evidence match."""
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    tag = subprocess.check_output(
        ["git", "rev-parse", "accountpulse-v1.0-complete^{commit}"], cwd=REPO, text=True
    ).strip()
    base = subprocess.check_output(
        ["git", "merge-base", "HEAD", "accountpulse-v1.0-complete"], cwd=REPO, text=True
    ).strip()
    if branch != "analysis/postlock-diagnostics" or tag != BASE_COMMIT or base != BASE_COMMIT:
        raise RuntimeError(f"Branch/base gate failed: branch={branch}, tag={tag}, base={base}")
    actual = protected_hashes()
    if actual != EXPECTED_PROTECTED:
        raise RuntimeError(f"Protected hash gate failed: {actual}")
    required_diag1 = (
        REPO / "DIAGNOSTICS_MANIFEST.yaml",
        EVIDENCE / "maven_slice_metrics_postlock.parquet",
        EVIDENCE / "maven_attribution_drift_postlock.parquet",
        REPO / "reports/TrackA_Explainability.md",
    )
    missing = [str(path) for path in required_diag1 if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing DIAG1 artifacts: {missing}")
    historical = (REPO / "reports/TrackA_Explainability.md").read_text(encoding="utf-8")
    if "CUDA is unavailable" not in historical or "INFRASTRUCTURE_FAILURE" not in historical:
        raise RuntimeError("Historical DIAG1 GPU failure record is missing")
    return actual


def qualify_gpu() -> dict[str, Any]:
    """Qualify exactly one CUDA device and the WSL driver surface."""
    result = subprocess.run(
        [
            "/usr/lib/wsl/lib/nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA qualification failed or physical GPU count is not one")
    return {
        "torch_cuda_available": True,
        "torch_device_count": 1,
        "torch_device_name": torch.cuda.get_device_name(0),
        "nvidia_smi": result.stdout.strip(),
    }


def map_encoded_feature(encoded_feature: str) -> tuple[str, str]:
    """Map fitted ColumnTransformer output to a raw feature and stable group."""
    raw_feature = encoded_feature
    if encoded_feature not in PREASSIGN_NUMERIC:
        for candidate in sorted(PREASSIGN_CATEGORICAL, key=len, reverse=True):
            if encoded_feature.startswith(f"{candidate}_"):
                raw_feature = candidate
                break
    if raw_feature.startswith("product_prior_"):
        group = "product_history"
    elif raw_feature.startswith("sector_prior_"):
        group = "sector_history"
    elif raw_feature.startswith("account_prior_") or raw_feature in {
        "days_since_prior_opportunity",
        "account_missing",
    }:
        group = "account_history"
    elif raw_feature in {"product", "series", "sales_price"}:
        group = "product"
    elif raw_feature in {"engage_month", "engage_quarter"}:
        group = "calendar"
    else:
        group = "firmographics"
    return raw_feature, group


def _cuda_booster(model: Any) -> xgb.Booster:
    model.set_params(device="cuda:0")
    booster: xgb.Booster = model.get_booster()
    booster.set_param({"device": "cuda:0"})
    config = json.loads(booster.save_config())
    if config["learner"]["generic_param"]["device"] != "cuda:0":
        raise RuntimeError("Frozen booster did not accept device=cuda:0")
    return booster


def _dmatrix(encoded: np.ndarray, names: list[str] | None = None) -> xgb.DMatrix:
    return xgb.DMatrix(np.asarray(encoded, dtype=np.float32), feature_names=names)


def _native_predict(model: Any, encoded: np.ndarray, *, margin: bool = False) -> np.ndarray:
    booster = _cuda_booster(model)
    return np.asarray(booster.predict(_dmatrix(encoded), output_margin=margin))


def validate_shap_additivity(
    contribution: np.ndarray, margin: np.ndarray, tolerance: float = 2e-5
) -> float:
    """Validate bias + feature contributions against the frozen raw margin."""
    error = float(
        np.max(
            np.abs(
                contribution[:, :-1].sum(axis=1)
                + contribution[:, -1]
                - np.asarray(margin).reshape(-1)
            )
        )
    )
    if error > tolerance:
        raise RuntimeError(f"TreeSHAP additivity failed: {error} > {tolerance}")
    return error


def _b4_shap(
    scorer: FrozenTrackAScorer,
    policy: pd.DataFrame,
    locked: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    names = list(scorer.preprocessor.get_feature_names_out())
    policy_x = np.asarray(scorer.preprocessor.transform(policy), dtype=np.float32)
    locked_x = np.asarray(scorer.preprocessor.transform(locked), dtype=np.float32)
    booster = _cuda_booster(scorer.classifiers["B4_xgboost_cuda"])
    policy_dm = _dmatrix(policy_x, names)
    locked_dm = _dmatrix(locked_x, names)
    policy_margin = np.asarray(booster.predict(policy_dm, output_margin=True))
    locked_margin = np.asarray(booster.predict(locked_dm, output_margin=True))
    policy_contrib = np.asarray(booster.predict(policy_dm, pred_contribs=True))
    locked_contrib = np.asarray(booster.predict(locked_dm, pred_contribs=True))
    errors = {
        "policy": validate_shap_additivity(policy_contrib, policy_margin),
        "locked": validate_shap_additivity(locked_contrib, locked_margin),
    }
    mapping = pd.DataFrame(
        [
            {
                "encoded_feature": name,
                "raw_feature": map_encoded_feature(name)[0],
                "feature_group": map_encoded_feature(name)[1],
                "position": position,
            }
            for position, name in enumerate(names)
        ]
    )
    encoded_rows = []
    for item in mapping.itertuples(index=False):
        p = policy_contrib[:, item.position]
        locked_values = locked_contrib[:, item.position]
        encoded_rows.append(
            {
                "summary_level": "ENCODED_FEATURE",
                "encoded_feature": item.encoded_feature,
                "raw_feature": item.raw_feature,
                "feature_group": item.feature_group,
                "mean_abs_shap_policy": float(np.mean(np.abs(p))),
                "mean_abs_shap_locked": float(np.mean(np.abs(locked_values))),
                "signed_mean_shap_policy": float(np.mean(p)),
                "signed_mean_shap_locked": float(np.mean(locked_values)),
            }
        )
    encoded = pd.DataFrame(encoded_rows)
    aggregate_rows = []
    for level, field in (("RAW_FEATURE", "raw_feature"), ("FEATURE_GROUP", "feature_group")):
        for value, subset in mapping.groupby(field, observed=True):
            positions = subset["position"].to_numpy(int)
            p = policy_contrib[:, positions].sum(axis=1)
            current = locked_contrib[:, positions].sum(axis=1)
            aggregate_rows.append(
                {
                    "summary_level": level,
                    "encoded_feature": None,
                    "raw_feature": str(value) if field == "raw_feature" else None,
                    "feature_group": str(value) if field == "feature_group" else str(subset["feature_group"].iloc[0]),
                    "mean_abs_shap_policy": float(np.mean(np.abs(p))),
                    "mean_abs_shap_locked": float(np.mean(np.abs(current))),
                    "signed_mean_shap_policy": float(np.mean(p)),
                    "signed_mean_shap_locked": float(np.mean(current)),
                }
            )
    global_frame = pd.concat([encoded, pd.DataFrame(aggregate_rows)], ignore_index=True)
    global_frame["importance_rank_policy"] = global_frame.groupby("summary_level")["mean_abs_shap_policy"].rank(method="min", ascending=False).astype(int)
    global_frame["importance_rank_locked"] = global_frame.groupby("summary_level")["mean_abs_shap_locked"].rank(method="min", ascending=False).astype(int)
    global_frame["rank_change"] = global_frame["importance_rank_locked"] - global_frame["importance_rank_policy"]
    policy_total = global_frame.groupby("summary_level")["mean_abs_shap_policy"].transform("sum")
    locked_total = global_frame.groupby("summary_level")["mean_abs_shap_locked"].transform("sum")
    global_frame["normalized_policy_importance"] = global_frame["mean_abs_shap_policy"] / policy_total
    global_frame["normalized_locked_importance"] = global_frame["mean_abs_shap_locked"] / locked_total
    global_frame["normalized_attribution_drift"] = (
        global_frame["normalized_locked_importance"] - global_frame["normalized_policy_importance"]
    ).abs()
    drift = global_frame.copy()
    drift["top_feature_turnover"] = (
        drift["importance_rank_policy"].le(10) != drift["importance_rank_locked"].le(10)
    )
    arrays = {
        "locked_contrib": locked_contrib,
        "locked_margin": locked_margin,
        "policy_contrib": policy_contrib,
        "policy_margin": policy_margin,
    }
    return global_frame, drift, errors, arrays


def _tie_break(score: np.ndarray, identifiers: pd.Series) -> np.ndarray:
    hashes = np.array(
        [int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:8], "big") for value in identifiers],
        dtype=np.float64,
    )
    hashes /= np.iinfo(np.uint64).max
    return np.asarray(score, dtype=float) + hashes * max(1.0, float(np.max(np.abs(score)))) * 1e-12


def value_capture_10(score: np.ndarray, value: np.ndarray, identifiers: pd.Series) -> float:
    """Frozen 10% value-capture definition with the A1 deterministic tie break."""
    stable = _tie_break(score, identifiers)
    count = max(1, math.ceil(0.10 * len(stable)))
    selected = np.argsort(-stable, kind="stable")[:count]
    return float(value[selected].sum() / max(value.sum(), 1e-12))


def _ranks(score: pd.Series, identifiers: pd.Series) -> np.ndarray:
    stable = _tie_break(score.to_numpy(float), identifiers)
    order = np.argsort(-stable, kind="stable")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def _local_shap(
    frame: pd.DataFrame,
    global_arrays: dict[str, np.ndarray],
    feature_names: list[str],
) -> pd.DataFrame:
    errors = pd.read_parquet(EVIDENCE / "maven_error_cases_postlock.parquet")
    missed = errors[errors["case_type"].eq("MISSED_WHALES")].sort_values("actual_value", ascending=False).drop_duplicates("opportunity_id").head(20)
    false = errors[errors["case_type"].eq("FALSE_PRIORITIES")].drop_duplicates("opportunity_id").head(20)
    ranks = pd.DataFrame(
        {
            "opportunity_id": frame["opportunity_id"],
            "B1_rank": _ranks(frame["B1_business_heuristic"], frame["opportunity_id"]),
            "AP-EV_rank": _ranks(frame["AccountPulse_EV"], frame["opportunity_id"]),
            "B4_rank": _ranks(frame["B4_xgboost_cuda"], frame["opportunity_id"]),
        }
    )
    disagreements = ranks.assign(
        rank_gap=lambda data: (data["B1_rank"] - data["AP-EV_rank"]).abs()
    ).nlargest(20, "rank_gap")
    targets = pd.concat(
        [
            missed[["opportunity_id"]].assign(case_type="MISSED_WHALE"),
            false[["opportunity_id"]].assign(case_type="FALSE_PRIORITY"),
            disagreements[["opportunity_id"]].assign(case_type="LARGEST_B1_APEV_DISAGREEMENT"),
        ],
        ignore_index=True,
    ).drop_duplicates(["opportunity_id", "case_type"])
    lookup = frame.reset_index().set_index("opportunity_id")
    rank_lookup = ranks.set_index("opportunity_id")
    contributions = global_arrays["locked_contrib"]
    margins = global_arrays["locked_margin"]
    rows = []
    for target in targets.itertuples(index=False):
        if target.opportunity_id not in lookup.index:
            continue
        item = lookup.loc[target.opportunity_id]
        position = int(item["index"])
        shap_values = contributions[position, :-1]
        positive = np.argsort(-shap_values)[:5]
        negative = np.argsort(shap_values)[:5]
        for direction, selected in (("POSITIVE", positive), ("NEGATIVE", negative)):
            for contribution_rank, encoded_position in enumerate(selected, start=1):
                encoded = feature_names[encoded_position]
                raw, group = map_encoded_feature(encoded)
                rows.append(
                    {
                        "opportunity_id": target.opportunity_id,
                        "case_type": target.case_type,
                        "actual_outcome": item["terminal_state"],
                        "actual_value": float(item["realized_value"]),
                        "B1_rank": int(rank_lookup.loc[target.opportunity_id, "B1_rank"]),
                        "AP-EV_rank": int(rank_lookup.loc[target.opportunity_id, "AP-EV_rank"]),
                        "B4_rank": int(rank_lookup.loc[target.opportunity_id, "B4_rank"]),
                        "model_margin": float(margins[position]),
                        "predicted_probability": float(expit(margins[position])),
                        "bias": float(contributions[position, -1]),
                        "contribution_direction": direction,
                        "contribution_rank": contribution_rank,
                        "encoded_feature": encoded,
                        "raw_feature": raw,
                        "feature_group": group,
                        "shap_value": float(shap_values[encoded_position]),
                        "interpretation": "MODEL_ATTRIBUTION_NOT_CAUSAL_EFFECT",
                    }
                )
    return pd.DataFrame(rows)


def _apev_components(
    scorer: FrozenTrackAScorer,
    frame: pd.DataFrame,
    *,
    validate_stored: bool = True,
    build_rows: bool = True,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    encoded = np.asarray(scorer.preprocessor.transform(frame), dtype=np.float32)
    timing = _native_predict(scorer.timing_model, encoded)[:, 1:3]
    win_columns, value_columns = [], []
    for time_bin in (1, 2):
        augmented = np.column_stack([encoded, np.full(len(encoded), time_bin, dtype=np.float32)])
        raw_win = _native_predict(scorer.conditional_win, augmented)
        win_columns.append(
            _calibrate(raw_win, scorer.selected_config["win"], scorer.conditional_calibrators)
        )
        raw_value = _native_predict(scorer.value_models["xgboost"], augmented)
        value_columns.append(np.maximum(np.expm1(raw_value), 0.0))
    win = np.column_stack(win_columns)
    value = np.column_stack(value_columns)
    days = np.array([15.0, 45.0])
    discount = np.exp(-float(scorer.selected_config["rho"]) * days)
    undiscounted = timing * win * value
    discounted = undiscounted * discount
    reconstructed = discounted.sum(axis=1)
    stored = frame["AccountPulse_EV"].to_numpy(float) if validate_stored else reconstructed.copy()
    maximum_error = float(np.max(np.abs(reconstructed - stored)))
    if validate_stored and maximum_error > 1e-4:
        raise RuntimeError(f"AP-EV component reconstruction failed: {maximum_error}")
    rows = []
    if build_rows:
        for index, opportunity_id in enumerate(frame["opportunity_id"]):
            for bin_position, (time_bin, terminal_days) in enumerate(((1, 15), (2, 45))):
                rows.append(
                    {
                        "record_type": "OPPORTUNITY_TIME_BIN",
                        "opportunity_id": opportunity_id,
                        "time_bin": time_bin,
                        "terminal_days": terminal_days,
                        "timing_probability": float(timing[index, bin_position]),
                        "conditional_win_probability": float(win[index, bin_position]),
                        "conditional_value": float(value[index, bin_position]),
                        "discount_weight": float(discount[bin_position]),
                        "undiscounted_bin_contribution": float(undiscounted[index, bin_position]),
                        "per_bin_ev_contribution": float(discounted[index, bin_position]),
                        "undiscounted_total": float(undiscounted[index].sum()),
                        "discounted_total": float(reconstructed[index]),
                        "final_accountpulse_ev_score": float(stored[index]),
                        "reconstruction_error": float(reconstructed[index] - stored[index]),
                        "rho": float(scorer.selected_config["rho"]),
                    }
                )
    arrays = {
        "timing": timing,
        "win": win,
        "value": value,
        "undiscounted": undiscounted,
        "discounted": discounted,
        "score": reconstructed,
        "maximum_reconstruction_error": np.array([maximum_error]),
    }
    return pd.DataFrame(rows), arrays


def _component_summary(frame: pd.DataFrame, arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    data = frame.copy()
    epsilon = 1e-12
    data["undiscounted_total"] = arrays["undiscounted"].sum(axis=1)
    data["discounted_total"] = arrays["discounted"].sum(axis=1)
    data["absolute_discount_penalty"] = (
        data["undiscounted_total"] - data["discounted_total"]
    )
    denominator = np.maximum(data["undiscounted_total"].to_numpy(float), epsilon)
    data["relative_discount_penalty"] = (
        data["absolute_discount_penalty"].to_numpy(float) / denominator
    )
    data["effective_discount_factor"] = data["discounted_total"].to_numpy(float) / denominator
    data["expected_component_days"] = (
        arrays["undiscounted"] * np.array([15.0, 45.0])
    ).sum(axis=1) / denominator
    data["conditional_expected_value"] = data["expected_value_given_win"].to_numpy(float)
    data["sales_price_quartile"] = pd.qcut(
        data["sales_price"], 4, labels=["Q1", "Q2", "Q3", "Q4"]
    ).astype(str)
    realized_cutpoints = data["realized_value"].quantile([0.25, 0.50, 0.75]).to_numpy(float)
    data["realized_value_quartile"] = np.select(
        [
            data["realized_value"].le(realized_cutpoints[0]),
            data["realized_value"].le(realized_cutpoints[1]),
            data["realized_value"].le(realized_cutpoints[2]),
        ],
        ["Q1", "Q2", "Q3"],
        default="Q4",
    )
    b1_selected = _ranks(data["B1_business_heuristic"], data["opportunity_id"]) <= math.ceil(0.1 * len(data))
    ap_selected = _ranks(data["AccountPulse_EV"], data["opportunity_id"]) <= math.ceil(0.1 * len(data))
    data["disagreement_group"] = np.select(
        [b1_selected & ap_selected, b1_selected & ~ap_selected, ~b1_selected & ap_selected],
        ["B1_AND_APEV", "B1_ONLY", "APEV_ONLY"],
        default="NEITHER",
    )
    errors = pd.read_parquet(EVIDENCE / "maven_error_cases_postlock.parquet")
    missed = set(errors.loc[errors["case_type"].eq("MISSED_WHALES"), "opportunity_id"].astype(str))
    data["missed_whale_status"] = np.where(data["opportunity_id"].astype(str).isin(missed), "MISSED_WHALE", "OTHER")
    opportunity_columns = [
        "opportunity_id",
        "undiscounted_total",
        "discounted_total",
        "absolute_discount_penalty",
        "relative_discount_penalty",
        "effective_discount_factor",
        "expected_component_days",
        "conditional_expected_value",
    ]
    opportunity_rows = data[opportunity_columns].copy()
    opportunity_rows.insert(0, "record_type", "OPPORTUNITY_DISCOUNT_SUMMARY")
    opportunity_rows["normalization_epsilon"] = epsilon
    rows: list[dict[str, Any]] = []
    for dimension in (
        "sales_price_quartile",
        "realized_value_quartile",
        "product",
        "sector",
        "disagreement_group",
        "missed_whale_status",
    ):
        for category, part in data.groupby(dimension, observed=True):
            rows.append(
                {
                    "record_type": "SLICE_COMPONENT_SUMMARY",
                    "summary_dimension": dimension,
                    "summary_value": str(category),
                    "support_n": len(part),
                    "mean_undiscounted_total": float(part["undiscounted_total"].mean()),
                    "mean_discounted_total": float(part["discounted_total"].mean()),
                    "mean_absolute_discount_penalty": float(
                        part["absolute_discount_penalty"].mean()
                    ),
                    "median_absolute_discount_penalty": float(
                        part["absolute_discount_penalty"].median()
                    ),
                    "mean_relative_discount_penalty": float(
                        part["relative_discount_penalty"].mean()
                    ),
                    "median_relative_discount_penalty": float(
                        part["relative_discount_penalty"].median()
                    ),
                    "mean_effective_discount_factor": float(
                        part["effective_discount_factor"].mean()
                    ),
                    "mean_expected_component_days": float(part["expected_component_days"].mean()),
                    "mean_sales_price": float(part["sales_price"].mean()),
                    "mean_actual_value": float(part["realized_value"].mean()),
                    "mean_conditional_expected_value": float(
                        part["conditional_expected_value"].mean()
                    ),
                }
            )
    comparators = (
        "sales_price",
        "realized_value",
        "expected_component_days",
        "undiscounted_total",
        "conditional_expected_value",
    )
    correlation_columns = [
        "absolute_discount_penalty",
        "relative_discount_penalty",
        *comparators,
    ]
    correlations = data[correlation_columns].corr(method="spearman")
    for penalty in ("absolute_discount_penalty", "relative_discount_penalty"):
        for comparator in comparators:
            rows.append(
                {
                    "record_type": "DISCOUNT_RELATIONSHIP",
                    "penalty_measure": penalty,
                    "summary_dimension": comparator,
                    "summary_value": f"SPEARMAN_WITH_{penalty.upper()}",
                    "support_n": len(data),
                    "spearman_correlation": float(correlations.loc[penalty, comparator]),
                }
            )
    return pd.concat([opportunity_rows, pd.DataFrame(rows)], ignore_index=True)


def _b1_concentration_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Contextualize B1 product/sector concentration against the LOCKED population."""
    selected_mask = _ranks(frame["B1_business_heuristic"], frame["opportunity_id"]) <= math.ceil(
        0.1 * len(frame)
    )
    selected = frame.loc[selected_mask]
    rows = []
    for dimension in ("product", "sector"):
        population_counts = frame.groupby(dimension, observed=True).size()
        selection_counts = selected.groupby(dimension, observed=True).size()
        population_values = frame.groupby(dimension, observed=True)["realized_value"].sum()
        selected_values = selected.groupby(dimension, observed=True)["realized_value"].sum()
        population_share = population_counts / len(frame)
        selection_share = selection_counts.reindex(population_counts.index, fill_value=0) / len(selected)
        population_value_share = population_values / max(frame["realized_value"].sum(), 1e-12)
        selected_value_share = selected_values.reindex(population_counts.index, fill_value=0) / max(
            selected["realized_value"].sum(), 1e-12
        )
        population_hhi = float(np.square(population_share).sum())
        selection_hhi = float(np.square(selection_share).sum())
        for category in population_counts.index:
            rows.append(
                {
                    "dimension": dimension,
                    "category": str(category),
                    "population_count": int(population_counts.loc[category]),
                    "selected_count": int(selection_counts.get(category, 0)),
                    "population_share": float(population_share.loc[category]),
                    "selection_share": float(selection_share.loc[category]),
                    "selection_lift": float(
                        selection_share.loc[category] / population_share.loc[category]
                    ),
                    "population_value_share": float(population_value_share.loc[category]),
                    "selected_value_share": float(selected_value_share.loc[category]),
                    "population_hhi": population_hhi,
                    "selection_hhi": selection_hhi,
                }
            )
    return pd.DataFrame(rows)


def _score_b4(scorer: FrozenTrackAScorer, frame: pd.DataFrame) -> np.ndarray:
    encoded = np.asarray(scorer.preprocessor.transform(frame), dtype=np.float32)
    margin = _native_predict(scorer.classifiers["B4_xgboost_cuda"], encoded, margin=True)
    return expit(margin)


def _score_apev(scorer: FrozenTrackAScorer, frame: pd.DataFrame) -> np.ndarray:
    return _apev_components(
        scorer, frame, validate_stored=False, build_rows=False
    )[1]["score"]


def _grouped_permutation(
    scorer: FrozenTrackAScorer, frame: pd.DataFrame, repeats: int = 100
) -> pd.DataFrame:
    group_columns = {
        group: [
            feature
            for feature in PREASSIGN_CATEGORICAL + PREASSIGN_NUMERIC
            if map_encoded_feature(feature)[1] == group
        ]
        for group in FEATURE_GROUPS
    }
    value = frame["realized_value"].to_numpy(float)
    identifiers = frame["opportunity_id"]
    baseline_scores = {
        "B4 XGBoost": frame["B4_xgboost_cuda"].to_numpy(float),
        "AP-EV": frame["AccountPulse_EV"].to_numpy(float),
    }
    baseline = {
        model: value_capture_10(score, value, identifiers) for model, score in baseline_scores.items()
    }
    rows = []
    for group_index, (group, columns) in enumerate(group_columns.items()):
        rng = np.random.default_rng(20260923 + group_index)
        permuted_frames = []
        for repeat in range(repeats):
            permutation = rng.permutation(len(frame))
            candidate = frame.copy()
            candidate.loc[:, columns] = frame.iloc[permutation][columns].to_numpy()
            candidate["_repeat"] = repeat
            permuted_frames.append(candidate)
        batch = pd.concat(permuted_frames, ignore_index=True)
        b4_scores = _score_b4(scorer, batch)
        apev_scores = _score_apev(scorer, batch)
        for model, scores in (("B4 XGBoost", b4_scores), ("AP-EV", apev_scores)):
            captures = []
            for repeat in range(repeats):
                start, end = repeat * len(frame), (repeat + 1) * len(frame)
                captures.append(value_capture_10(scores[start:end], value, identifiers))
            captures_array = np.asarray(captures)
            rows.append(
                {
                    "model": model,
                    "feature_group": group,
                    "features": json.dumps(columns),
                    "repeats": repeats,
                    "seed": 20260923 + group_index,
                    "baseline_value_capture_10": baseline[model],
                    "permuted_mean_value_capture_10": float(captures_array.mean()),
                    "delta": float(baseline[model] - captures_array.mean()),
                    "permutation_standard_deviation": float(captures_array.std(ddof=1)),
                    "permutation_q025": float(np.quantile(captures_array, 0.025)),
                    "permutation_q975": float(np.quantile(captures_array, 0.975)),
                    "permutation_distribution_interval_95": (
                        f"[{np.quantile(captures_array, 0.025):.10g}, "
                        f"{np.quantile(captures_array, 0.975):.10g}]"
                    ),
                    "uncertainty_scope": (
                        "PERTURBATION_VARIABILITY_CONDITIONAL_ON_FIXED_LOCKED_COHORT_"
                        "AND_FROZEN_MODEL_NOT_SAMPLING_CONFIDENCE_INTERVAL"
                    ),
                    "interpretation_limit": "CORRELATED_INFORMATION_ACROSS_GROUPS_CAN_ATTENUATE_IMPORTANCE",
                }
            )
    return pd.DataFrame(rows)


def empirical_grid(series: pd.Series, points: int = 15) -> np.ndarray:
    """Deterministic grid bounded by empirical fifth and 95th percentiles."""
    low, high = series.quantile([0.05, 0.95]).to_numpy(float)
    return np.linspace(low, high, points)


def _pdp_ice(scorer: FrozenTrackAScorer, frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260923)
    sample_positions = np.sort(rng.choice(len(frame), min(50, len(frame)), replace=False))
    sample = frame.iloc[sample_positions].copy()
    rows = []
    figure_dir = REPO / "artifacts/figures/tracka_gpu_explainability"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for feature in PDP_FEATURES:
        grid = empirical_grid(frame[feature])
        pdp_frames, ice_frames = [], []
        for grid_index, value in enumerate(grid):
            population = frame.copy()
            population[feature] = value
            population["_grid_index"] = grid_index
            pdp_frames.append(population)
            ice = sample.copy()
            ice[feature] = value
            ice["_grid_index"] = grid_index
            ice_frames.append(ice)
        population_scores = _score_b4(scorer, pd.concat(pdp_frames, ignore_index=True))
        ice_scores = _score_b4(scorer, pd.concat(ice_frames, ignore_index=True))
        feature_rows = []
        for grid_index, value in enumerate(grid):
            pop = population_scores[grid_index * len(frame) : (grid_index + 1) * len(frame)]
            ice = ice_scores[grid_index * len(sample) : (grid_index + 1) * len(sample)]
            row = {
                "feature": feature,
                "grid_value": float(value),
                "grid_lower_bound": float(grid[0]),
                "grid_upper_bound": float(grid[-1]),
                "mean_prediction": float(pop.mean()),
                "ice_q10": float(np.quantile(ice, 0.10)),
                "ice_q25": float(np.quantile(ice, 0.25)),
                "ice_q50": float(np.quantile(ice, 0.50)),
                "ice_q75": float(np.quantile(ice, 0.75)),
                "ice_q90": float(np.quantile(ice, 0.90)),
                "sample_count": len(sample),
                "interpretation": "MODEL RESPONSE, NOT CAUSAL EFFECT",
                "warning": "Correlated-feature perturbations may generate unrealistic combinations.",
            }
            rows.append(row)
            feature_rows.append(row)
        plot = pd.DataFrame(feature_rows)
        fig, axis = plt.subplots(figsize=(7, 4.5))
        axis.plot(plot["grid_value"], plot["mean_prediction"], label="mean response")
        axis.fill_between(plot["grid_value"], plot["ice_q10"], plot["ice_q90"], alpha=0.2, label="ICE 10-90%")
        axis.set(xlabel=feature, ylabel="B4 predicted probability", title=f"B4 {feature}: MODEL RESPONSE, NOT CAUSAL EFFECT")
        axis.legend()
        fig.text(0.5, 0.025, "Correlated-feature perturbations may generate unrealistic combinations.", ha="center", fontsize=7)
        fig.text(0.5, 0.005, f"base_protocol={BASE_PROTOCOL} | supplement_protocol={SUPPLEMENT_PROTOCOL} | {LABEL} | post_lock=true | decision_eligible=false | infrastructure_recovery=true | A1=RETAIN_BASELINE", ha="center", fontsize=5)
        fig.tight_layout(rect=(0, 0.055, 1, 1))
        fig.savefig(figure_dir / f"b4_pdp_ice_{feature}.png", dpi=160)
        plt.close(fig)
    return pd.DataFrame(rows)


def _write_global_figure(global_frame: pd.DataFrame) -> None:
    figure_dir = REPO / "artifacts/figures/tracka_gpu_explainability"
    figure_dir.mkdir(parents=True, exist_ok=True)
    groups = global_frame[global_frame["summary_level"].eq("FEATURE_GROUP")].sort_values("mean_abs_shap_locked")
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.barh(groups["feature_group"], groups["mean_abs_shap_policy"], alpha=0.7, label="POLICY")
    axis.barh(groups["feature_group"], groups["mean_abs_shap_locked"], alpha=0.7, label="LOCKED")
    axis.set(xlabel="mean absolute SHAP (margin)", title="B4 grouped native TreeSHAP — diagnostic only")
    axis.legend()
    fig.text(0.5, 0.025, "Attribution is not a causal effect.", ha="center", fontsize=7)
    fig.text(0.5, 0.005, f"base_protocol={BASE_PROTOCOL} | supplement_protocol={SUPPLEMENT_PROTOCOL} | {LABEL} | post_lock=true | decision_eligible=false | infrastructure_recovery=true | A1=RETAIN_BASELINE", ha="center", fontsize=5)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(figure_dir / "b4_grouped_shap_policy_locked.png", dpi=160)
    plt.close(fig)


def _replace_section(path: Path, heading: str, content: str) -> None:
    text = path.read_text(encoding="utf-8")
    marker = f"## {heading}"
    if marker in text:
        text = text.split(marker, maxsplit=1)[0].rstrip() + "\n"
    path.write_text(text.rstrip() + f"\n\n{marker}\n\n{content.strip()}\n", encoding="utf-8")


def _markdown(frame: pd.DataFrame, columns: list[str], rows: int = 10) -> str:
    data = frame[[column for column in columns if column in frame]].head(rows)
    names = [str(column) for column in data.columns]
    output = ["| " + " | ".join(names) + " |", "|" + "|".join(["---"] * len(names)) + "|"]
    for values in data.itertuples(index=False, name=None):
        output.append("| " + " | ".join(f"{value:.4f}" if isinstance(value, float) else str(value) for value in values) + " |")
    return "\n".join(output)


def _write_reports(
    global_frame: pd.DataFrame,
    local: pd.DataFrame,
    components: pd.DataFrame,
    permutation: pd.DataFrame,
    pdp: pd.DataFrame,
    concentration: pd.DataFrame,
    gpu: dict[str, Any],
    additivity: dict[str, Any],
) -> None:
    groups = global_frame[global_frame["summary_level"].eq("FEATURE_GROUP")].sort_values("mean_abs_shap_locked", ascending=False)
    raw = global_frame[global_frame["summary_level"].eq("RAW_FEATURE")].sort_values("mean_abs_shap_locked", ascending=False)
    relationships = components[components["record_type"].eq("DISCOUNT_RELATIONSHIP")]
    quartiles = components[
        components["record_type"].eq("SLICE_COMPONENT_SUMMARY")
        & components["summary_dimension"].isin(
            ["sales_price_quartile", "realized_value_quartile"]
        )
    ]
    perm = permutation.sort_values(["model", "delta"], ascending=[True, False])
    concentration_hhi = (
        concentration.groupby("dimension", observed=True)
        .agg(population_hhi=("population_hhi", "first"), selection_hhi=("selection_hhi", "first"))
        .reset_index()
    )
    concentration_display = concentration[concentration["selected_count"].gt(0)].sort_values(
        ["dimension", "selection_lift"], ascending=[True, False]
    )
    relationship_lookup = relationships.set_index(["penalty_measure", "summary_dimension"])[
        "spearman_correlation"
    ]
    absolute_sales_rho = float(
        relationship_lookup.loc[("absolute_discount_penalty", "sales_price")]
    )
    relative_sales_rho = float(
        relationship_lookup.loc[("relative_discount_penalty", "sales_price")]
    )
    relative_days_rho = float(
        relationship_lookup.loc[("relative_discount_penalty", "expected_component_days")]
    )
    relative_price_conclusion = (
        "The near-zero normalized association does not support a claim that high-price "
        "opportunities were disproportionately penalized."
        if abs(relative_sales_rho) < 0.10
        else "The normalized association should be considered before making any claim about "
        "disproportionate penalty by price."
    )
    hhi_lookup = concentration_hhi.set_index("dimension")
    product_population_hhi = float(hhi_lookup.loc["product", "population_hhi"])
    product_selection_hhi = float(hhi_lookup.loc["product", "selection_hhi"])
    sector_population_hhi = float(hhi_lookup.loc["sector", "population_hhi"])
    sector_selection_hhi = float(hhi_lookup.loc["sector", "selection_hhi"])
    metadata = f"""base_protocol: `{BASE_PROTOCOL}`
diagnostic_protocol: `{DIAGNOSTIC_PROTOCOL}`
supplement_protocol: `{SUPPLEMENT_PROTOCOL}`
result_label: `{LABEL}`
post_lock: `true`
decision_eligible: `false`
infrastructure_recovery: `true`
a1_decision: `RETAIN_BASELINE`"""
    supplement = f"""# Track A GPU Explainability Supplement

{metadata}

DIAG1 originally and correctly recorded `NOT_EVALUABLE_INFRASTRUCTURE_FAILURE_CUDA_UNAVAILABLE`. GPU capability recovered later; this GPU1 supplement used only frozen, read-only inference with every XGBoost booster set to `device=cuda:0`. No fitting, training, updating, calibration selection, or scientific retuning occurred.

## GPU qualification and TreeSHAP

- Device: {gpu['torch_device_name']}; physical device count: {gpu['torch_device_count']}.
- Native B4 TreeSHAP additivity maximum errors: POLICY `{additivity['policy']:.3g}`, LOCKED `{additivity['locked']:.3g}`.
- Contributions explain frozen B4 model margins, not causal effects.

## Global grouped attribution

{_markdown(groups, ['feature_group', 'mean_abs_shap_policy', 'mean_abs_shap_locked', 'importance_rank_policy', 'importance_rank_locked', 'normalized_attribution_drift'], 6)}

## Top LOCKED raw features

{_markdown(raw, ['raw_feature', 'feature_group', 'mean_abs_shap_locked', 'importance_rank_locked', 'rank_change'], 12)}

## Local predefined cases

Local explanations cover only predefined missed whales, false priorities, and largest frozen B1/AP-EV rank disagreements. The tidy artifact contains the five largest positive and negative margin contributions per case. Nothing was selected after inspecting SHAP values.

{_markdown(local.sort_values('actual_value', ascending=False), ['case_type', 'opportunity_id', 'actual_outcome', 'actual_value', 'encoded_feature', 'shap_value'], 12)}

## AP-EV components and discount behavior

The exact two-bin frozen computation was reconstructed with `rho=0.005`; the stored score was not changed. The earlier finding—full ValueCapture@10% 24.23% versus 25.32% without discount—remains explanatory only.

### Absolute-value scaling versus proportional timing

The absolute discount penalty has Spearman rho `{absolute_sales_rho:.4f}` with sales price. Because the absolute penalty is denominated in value units, that relationship includes opportunity-value scale. The normalized relative penalty has sales-price rho `{relative_sales_rho:.4f}` and expected-component-days rho `{relative_days_rho:.4f}`. {relative_price_conclusion} The perfect monotone relationship with expected component days reflects the fixed positive `rho` applied to the two-bin timing mixture; it identifies timing-associated proportional discounting, not a causal effect.

{_markdown(relationships, ['penalty_measure', 'summary_dimension', 'spearman_correlation'], 10)}

The opportunity-level artifact records `absolute_discount_penalty`, `relative_discount_penalty`, `effective_discount_factor`, `expected_component_days`, and the directly available stored `conditional_expected_value` for every LOCKED opportunity. The denominator is `max(undiscounted_total, 1e-12)`. Realized-value quartile cutpoints retain ties, so empty middle quartiles are possible when zero realized values dominate.

{_markdown(quartiles, ['summary_dimension', 'summary_value', 'support_n', 'mean_absolute_discount_penalty', 'mean_relative_discount_penalty', 'mean_expected_component_days'], 8)}

## Grouped decision-specific permutation

{_markdown(perm, ['model', 'feature_group', 'baseline_value_capture_10', 'permuted_mean_value_capture_10', 'delta', 'permutation_standard_deviation', 'permutation_q025', 'permutation_q975'], 12)}

`permutation_q025` and `permutation_q975` form the `permutation_distribution_interval_95`: empirical 2.5th and 97.5th percentiles across repeated perturbations. This interval describes permutation variability conditional on the fixed LOCKED cohort and frozen model. It is **not a sampling confidence interval**.

Correlated information across feature groups can attenuate grouped permutation importance. These results were not used for feature selection.

## B1 concentration context

The B1 product/sector concentration results are contextualized against the underlying LOCKED Maven opportunity mix. `selection_lift` is selection share divided by population share; value shares use observed realized value. This is concentration analysis, not fairness analysis.

Product HHI rises from `{product_population_hhi:.4f}` in the LOCKED population to `{product_selection_hhi:.4f}` in B1's selected set, so B1 product concentration is substantially beyond the underlying product mix. Sector HHI rises more modestly, from `{sector_population_hhi:.4f}` to `{sector_selection_hhi:.4f}`. Category-level lifts below show where those differences arise.

{_markdown(concentration_hhi, ['dimension', 'population_hhi', 'selection_hhi'], 2)}

{_markdown(concentration_display, ['dimension', 'category', 'population_share', 'selection_share', 'selection_lift', 'population_value_share', 'selected_value_share'], 20)}

## PDP / ICE

PDP/ICE was restricted to the five predefined inputs over empirical 5th–95th percentile grids with a deterministic 50-row ICE sample. Every result is **MODEL RESPONSE, NOT CAUSAL EFFECT**. Correlated-feature perturbations may generate unrealistic combinations.

{_markdown(pdp.groupby('feature').agg(mean_response=('mean_prediction', 'mean'), minimum=('mean_prediction', 'min'), maximum=('mean_prediction', 'max')).reset_index(), ['feature', 'mean_response', 'minimum', 'maximum'], 5)}

## Optional component SHAP

Timing multiclass and conditional-model SHAP were not collapsed into a single attribution because class/time-bin and augmented-time semantics would be ambiguous. Status: `NOT_EVALUABLE_METHOD_AMBIGUITY`. This did not block required GPU1 analyses.

## Scientific boundary

A1 remains `RETAIN_BASELINE`. All results are `POST_LOCK_DIAGNOSTIC_ONLY`, decision-ineligible, non-causal, and suitable only for prospective A2 hypotheses.
"""
    (REPO / "reports/TrackA_GPU_Explainability_Supplement.md").write_text(supplement, encoding="utf-8")
    explainability = f"""DIAG1 initially could not execute GPU-dependent explainability and its `INFRASTRUCTURE_FAILURE` record above remains unchanged. Later WSL/PyTorch CUDA qualification succeeded on `{gpu['torch_device_name']}`. GPU1 used only frozen read-only inference with `device=cuda:0`; no model training, updating, recalibration, feature selection, or scientific retuning occurred.

Native B4 TreeSHAP passed strict additivity on POLICY and LOCKED. See [the separate GPU supplement](TrackA_GPU_Explainability_Supplement.md). All attributions are model explanations, not causal effects."""
    _replace_section(REPO / "reports/TrackA_Explainability.md", "GPU Infrastructure-Recovery Supplement", explainability)
    diagnostics = f"""GPU capability recovered after DIAG1. Frozen B4 native TreeSHAP, exact AP-EV component reconstruction, grouped permutation, and bounded PDP/ICE were completed under `{SUPPLEMENT_PROTOCOL}`. This does not overwrite DIAG1's original infrastructure failure or change A1. Top LOCKED attribution groups were: {', '.join(groups.head(3)['feature_group'])}."""
    _replace_section(REPO / "reports/TrackA_PostLock_Diagnostics.md", "GPU Infrastructure-Recovery Supplement", diagnostics)
    monitoring = """The completed GPU1 attribution baseline enables the following **PROPOSED A2, NOT A1-VALIDATED** monitoring: feature-attribution rank drift, normalized mean-|SHAP| drift, top-feature turnover, and grouped attribution drift. Monitor both encoded and raw/grouped levels, compare like-for-like frozen preprocessing, and investigate rather than automatically promote or retrain. None is a retroactive A1 gate."""
    _replace_section(REPO / "reports/TrackA_Operational_Monitoring.md", "GPU Infrastructure-Recovery Supplement", monitoring)


def run() -> dict[str, Any]:
    """Execute the separate GPU1 supplement after all immutability gates pass."""
    before = verify_immutability_gate()
    diag1_manifest_hash = _sha256(REPO / "DIAGNOSTICS_MANIFEST.yaml")
    gpu = qualify_gpu()
    scorer: FrozenTrackAScorer = joblib.load(BUNDLE_PATH)
    development = pd.read_parquet(EVIDENCE / "maven_development.parquet")
    policy = development[development["split"].eq("POLICY")].reset_index(drop=True)
    features = pd.read_parquet(EVIDENCE / "maven_locked_features.parquet")
    predictions = pd.read_parquet(EVIDENCE / "maven_locked_predictions.parquet")
    locked = features.merge(
        predictions, on=["opportunity_id", "account_key", "engage_date"], validate="1:1"
    ).reset_index(drop=True)
    global_frame, drift, additivity, arrays = _b4_shap(scorer, policy, locked)
    local = _local_shap(
        locked, arrays, list(scorer.preprocessor.get_feature_names_out())
    )
    component_detail, component_arrays = _apev_components(scorer, locked)
    component_summary = _component_summary(locked, component_arrays)
    components = pd.concat([component_detail, component_summary], ignore_index=True)
    permutation = _grouped_permutation(scorer, locked)
    pdp = _pdp_ice(scorer, locked)
    concentration = _b1_concentration_context(locked)
    _write_global_figure(global_frame)
    outputs = {
        "maven_b4_shap_global_gpu_postlock": global_frame,
        "maven_b4_shap_attribution_drift_gpu_postlock": drift,
        "maven_b4_shap_local_gpu_postlock": local,
        "maven_apev_components_gpu_postlock": components,
        "maven_grouped_permutation_gpu_postlock": permutation,
        "maven_b4_pdp_ice_gpu_postlock": pdp,
    }
    for name, frame in outputs.items():
        write_table(name, _meta(frame))
    _write_reports(
        global_frame,
        local,
        components,
        permutation,
        pdp,
        concentration,
        gpu,
        additivity,
    )
    after = protected_hashes()
    if before != after or after != EXPECTED_PROTECTED:
        raise RuntimeError(f"Protected files changed: before={before}, after={after}")
    locked_result = json.loads((EVIDENCE / "maven_locked_result.json").read_text(encoding="utf-8"))
    if locked_result["decision"] != "RETAIN_BASELINE":
        raise RuntimeError("A1 decision changed")
    artifacts = {
        f"artifacts/evidence/{name}.parquet": _sha256(EVIDENCE / f"{name}.parquet")
        for name in OUTPUT_NAMES
    }
    figures = {
        str(path.relative_to(REPO)): _sha256(path)
        for path in sorted((REPO / "artifacts/figures/tracka_gpu_explainability").glob("*.png"))
    }
    reports = {
        str(path.relative_to(REPO)): _sha256(path)
        for path in (
            REPO / "reports/TrackA_GPU_Explainability_Supplement.md",
            REPO / "reports/TrackA_Explainability.md",
            REPO / "reports/TrackA_PostLock_Diagnostics.md",
            REPO / "reports/TrackA_Operational_Monitoring.md",
        )
    }
    manifest = {
        "base_protocol": BASE_PROTOCOL,
        "diagnostic_protocol": DIAGNOSTIC_PROTOCOL,
        "supplement_protocol": SUPPLEMENT_PROTOCOL,
        "result_label": LABEL,
        "post_lock": True,
        "decision_eligible": False,
        "infrastructure_recovery": True,
        "a1_decision": "RETAIN_BASELINE",
        "branch": "analysis/postlock-diagnostics",
        "base_commit": BASE_COMMIT,
        "diag1_historical_manifest_sha256": diag1_manifest_hash,
        "diag1_initial_gpu_status_preserved": True,
        "gpu_qualification": gpu,
        "xgboost_device": "cuda:0",
        "b4_shap_additivity_max_error": additivity,
        "apev_max_reconstruction_error": float(component_arrays["maximum_reconstruction_error"][0]),
        "optional_component_shap": "NOT_EVALUABLE_METHOD_AMBIGUITY",
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "protected_byte_identical": before == after,
        "no_training_or_retuning": True,
        "artifacts": artifacts,
        "figures": figures,
        "reports": reports,
    }
    (REPO / "DIAGNOSTICS_GPU_SUPPLEMENT_MANIFEST.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return {
        "status": "ACCOUNT_PULSE_GPU_EXPLAINABILITY_SUPPLEMENT_STATUS",
        "gpu": gpu,
        "artifacts": len(outputs),
        "protected_byte_identical": True,
        "a1_decision": "RETAIN_BASELINE",
        "no_training_or_retuning": True,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
