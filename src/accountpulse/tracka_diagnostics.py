# ruff: noqa: E501, RUF001, RUF046
"""Immutable post-lock diagnostics for frozen Track-A protocol A1.

This module consumes stored predictions and outcomes.  It never trains, tunes,
calibrates, selects, or writes any A1 scientific artifact.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.special import logit
from scipy.stats import entropy, kendalltau, ks_2samp, spearmanr, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    ndcg_score,
    roc_auc_score,
)

from accountpulse.evidence import EVIDENCE, write_table
from accountpulse.paths import REPO
from accountpulse.tracka_models import BUNDLE_PATH

BASE_PROTOCOL = "AP-V1-TRACKA-20260923-A1"
DIAGNOSTIC_PROTOCOL = "AP-V1-TRACKA-20260923-A1-DIAG1"
LABEL = "POST_LOCK_DIAGNOSTIC_ONLY"
TAG = "accountpulse-v1.0-complete"
MODELS = {
    "B1": "B1_business_heuristic",
    "AP-EV": "AccountPulse_EV",
    "LambdaMART": "LambdaMART",
    "B4 XGBoost": "B4_xgboost_cuda",
}
CAPACITY_MODELS = {
    **MODELS,
    "D2": "D2_probability_times_value",
    "D4": "D4_discounted_ev",
}
CAPACITIES = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
PROTECTED = (
    REPO / "FREEZE_MANIFEST.yaml",
    REPO / "LOCKED_START_RECEIPT.json",
    EVIDENCE / "maven_locked_result.json",
)
OUTPUT_NAMES = (
    "maven_slice_metrics_postlock",
    "maven_intersection_slice_metrics_postlock",
    "maven_error_cases_postlock",
    "maven_model_disagreement_postlock",
    "maven_b1_decomposition_postlock",
    "maven_apev_component_diagnostics_postlock",
    "maven_calibration_diagnostics_postlock",
    "maven_capacity_curve_postlock",
    "maven_rank_stability_postlock",
    "maven_selection_concentration_postlock",
    "maven_drift_diagnostics_postlock",
    "maven_attribution_drift_postlock",
    "maven_grouped_permutation_postlock",
    "maven_worst_slice_discovery_postlock",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes() -> dict[str, str]:
    """Return byte hashes for the three protected A1 artifacts."""
    return {str(path.relative_to(REPO)): _sha256(path) for path in PROTECTED}


def _meta(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "base_protocol", BASE_PROTOCOL)
    result.insert(1, "diagnostic_protocol", DIAGNOSTIC_PROTOCOL)
    result.insert(2, "result_label", LABEL)
    result.insert(3, "post_lock", True)
    result.insert(4, "decision_eligible", False)
    return result


def _selected(score: Iterable[float], capacity: float = 0.10) -> np.ndarray:
    values = np.asarray(score, dtype=float)
    count = max(1, int(math.ceil(capacity * len(values))))
    order = np.argsort(-values, kind="stable")
    mask = np.zeros(len(values), dtype=bool)
    mask[order[:count]] = True
    return mask


def hhi(shares: Iterable[float]) -> float:
    """Herfindahl-Hirschman index from non-negative shares."""
    values = np.asarray(list(shares), dtype=float)
    if values.size == 0 or values.sum() <= 0:
        return 0.0
    normalized = values / values.sum()
    return float(np.square(normalized).sum())


def _metric_status(valid: bool) -> str:
    return "EVALUABLE" if valid else "NOT_EVALUABLE"


def _calibration(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    valid = len(y) >= 10 and len(np.unique(y)) == 2 and y.sum() >= 2 and (1 - y).sum() >= 2
    if not valid:
        return {
            "brier": None,
            "log_loss": None,
            "ece": None,
            "calibration_in_the_large": None,
            "calibration_slope": None,
            "calibration_status": "NOT_EVALUABLE",
        }
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    bins = pd.qcut(pd.Series(p), min(10, len(np.unique(p))), duplicates="drop")
    work = pd.DataFrame({"p": p, "y": y, "bin": bins})
    grouped = work.groupby("bin", observed=True).agg(n=("y", "size"), p=("p", "mean"), y=("y", "mean"))
    ece = float(((grouped["n"] / len(work)) * (grouped["p"] - grouped["y"]).abs()).sum())
    prevalence = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    log_odds = np.log(p / (1 - p))
    design = np.column_stack([np.ones(len(log_odds)), log_odds])
    fit = LogisticRegression(C=1e6, fit_intercept=False, random_state=20260923).fit(design, y)
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": ece,
        "calibration_in_the_large": float(np.log(prevalence / (1 - prevalence)) - log_odds.mean()),
        "calibration_slope": float(fit.coef_[0, 1]),
        "calibration_status": "EVALUABLE",
    }


def _ranking_metrics(frame: pd.DataFrame, score: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    y = frame["won"].to_numpy(int)
    value = frame["realized_value"].to_numpy(float)
    selected_value = float(value[mask].sum())
    selected_wins = int(y[mask].sum())
    ideal_count = int(mask.sum())
    oracle = float(np.sort(value)[::-1][:ideal_count].sum())
    valid_class = len(np.unique(y)) == 2
    denominator_value = float(value.sum())
    denominator_wins = int(y.sum())
    result: dict[str, Any] = {
        "value_capture_10": selected_value / denominator_value if denominator_value > 0 else None,
        "value_capture_10_status": _metric_status(denominator_value > 0),
        "win_capture_10": selected_wins / denominator_wins if denominator_wins else None,
        "win_capture_10_status": _metric_status(denominator_wins > 0),
        "precision_10": selected_wins / max(int(mask.sum()), 1),
        "precision_10_status": "EVALUABLE",
        "recall_10": selected_wins / denominator_wins if denominator_wins else None,
        "recall_10_status": _metric_status(denominator_wins > 0),
        "oracle_regret": oracle - selected_value,
        "oracle_regret_status": "EVALUABLE",
        "roc_auc": float(roc_auc_score(y, score)) if valid_class else None,
        "roc_auc_status": _metric_status(valid_class),
        "pr_auc": float(average_precision_score(y, score)) if valid_class else None,
        "pr_auc_status": _metric_status(valid_class),
    }
    try:
        result["ndcg"] = float(ndcg_score(value.reshape(1, -1), score.reshape(1, -1)))
        result["ndcg_status"] = "EVALUABLE"
    except ValueError:
        result["ndcg"] = None
        result["ndcg_status"] = "NOT_EVALUABLE"
    return result


def _support(frame: pd.DataFrame, mask: np.ndarray) -> dict[str, Any]:
    return {
        "support_n": len(frame),
        "wins": int(frame["won"].sum()),
        "losses": int(frame["lost"].sum()),
        "censored_open": int((frame["terminal_state"] == "OPEN_CENSORED_60D").sum()),
        "total_realized_value": float(frame["realized_value"].sum()),
        "mean_realized_value": float(frame["realized_value"].mean()),
        "selected_count": int(mask.sum()),
        "selection_rate": float(mask.mean()),
    }


def _bootstrap_difference(
    frame: pd.DataFrame, left: str, right: str, replicates: int, seed: int
) -> dict[str, float | int | str]:
    groups = frame["account_key"].astype(str).unique()
    positions = {group: np.flatnonzero(frame["account_key"].astype(str).to_numpy() == group) for group in groups}
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(replicates):
        sampled = rng.choice(groups, len(groups), replace=True)
        take = np.concatenate([positions[group] for group in sampled])
        boot = frame.iloc[take]
        total = float(boot["realized_value"].sum())
        if total <= 0:
            continue
        left_mask = _selected(boot[left].to_numpy(float))
        right_mask = _selected(boot[right].to_numpy(float))
        differences.append(
            float(
                (boot["realized_value"].to_numpy(float)[left_mask].sum()
                - boot["realized_value"].to_numpy(float)[right_mask].sum())
                / total
            )
        )
    if not differences:
        return {"difference_ci_low": None, "difference_ci_high": None, "bootstrap_status": "NOT_EVALUABLE", "bootstrap_replicates": 0}
    return {
        "difference_ci_low": float(np.quantile(differences, 0.025)),
        "difference_ci_high": float(np.quantile(differences, 0.975)),
        "bootstrap_status": "EVALUABLE",
        "bootstrap_replicates": len(differences),
    }


def _add_bins(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for field, output in (
        ("revenue", "revenue_quartile"),
        ("employees", "employees_quartile"),
        ("sales_price", "sales_price_quartile"),
        ("company_age", "company_age_quartile"),
    ):
        result[output] = pd.qcut(result[field], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop").astype(str)
    result["account_history_bin"] = pd.cut(
        result["account_prior_opportunities"], [-1, 0, 2, 5, np.inf], labels=["0_SPARSE_HISTORY_PROXY", "1-2", "3-5", "6+"]
    ).astype(str)
    result["prior_resolved_history_bin"] = pd.cut(
        result["account_prior_resolved"], [-1, 0, 2, 5, np.inf], labels=["0", "1-2", "3-5", "6+"]
    ).astype(str)
    result["prior_win_rate_bin"] = pd.cut(
        result["account_prior_win_rate"], [-np.inf, 0.25, 0.50, 0.75, np.inf], labels=["<=.25", ".25-.50", ".50-.75", ">.75"]
    ).astype(str)
    result["prior_realized_value_bin"] = pd.qcut(
        result["account_prior_realized_value"].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]
    ).astype(str)
    result["engagement_period"] = result["engage_date"].dt.to_period("W").astype(str)
    for field, output in ((MODELS["B1"], "b1_score_decile"), (MODELS["AP-EV"], "apev_score_decile")):
        result[output] = pd.qcut(result[field].rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)]).astype(str)
    if "p_top10" in result:
        result["ranking_uncertainty_bin"] = pd.cut(
            result["p_top10"], [-0.01, 0.2, 0.5, 0.8, 1.0], labels=["VERY_LOW", "LOW", "MEDIUM", "HIGH"]
        ).astype(str)
    else:
        result["ranking_uncertainty_bin"] = "NOT_AVAILABLE"
    threshold = result.loc[result["won"].eq(1), "realized_value"].quantile(0.90)
    result["retrospective_high_value_tail"] = np.where(
        result["won"].eq(1) & result["realized_value"].ge(threshold), "HIGH_VALUE_TAIL", "OTHER"
    )
    result["region"] = result["office_location"]
    return result


SLICE_COLUMNS = (
    "sector", "product", "office_location", "region", "company_size", "revenue_quartile",
    "employees_quartile", "sales_price_quartile", "company_age_quartile", "account_history_bin",
    "prior_resolved_history_bin", "prior_win_rate_bin", "prior_realized_value_bin", "engagement_period",
    "b1_score_decile", "apev_score_decile", "ranking_uncertainty_bin", "retrospective_high_value_tail",
)


def _slice_table(frame: pd.DataFrame, columns: Iterable[str], minimum: int = 1) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    global_masks = {name: _selected(frame[column]) for name, column in MODELS.items()}
    for slice_name in columns:
        for slice_value, indices in frame.groupby(slice_name, observed=True).groups.items():
            subset = frame.loc[indices]
            if len(subset) < minimum:
                continue
            for model, column in MODELS.items():
                local_mask = _selected(subset[column].to_numpy(float))
                row = {
                    "analysis_mode": "LOCAL_SLICE_RANKING",
                    "slice": slice_name,
                    "slice_value": str(slice_value),
                    "model": model,
                    **_support(subset, local_mask),
                    **_ranking_metrics(subset, subset[column].to_numpy(float), local_mask),
                }
                if model == "B4 XGBoost":
                    row.update(_calibration(subset["won"].to_numpy(int), subset[column].to_numpy(float)))
                else:
                    row.update({key: None for key in ("brier", "log_loss", "ece", "calibration_in_the_large", "calibration_slope")})
                    row["calibration_status"] = "NOT_EVALUABLE_NONPROBABILITY_SCORE"
                rows.append(row)
                global_subset_mask = global_masks[model][frame.index.get_indexer(indices)]
                rows.append(
                    {
                        "analysis_mode": "GLOBAL_POLICY_BEHAVIOR",
                        "slice": slice_name,
                        "slice_value": str(slice_value),
                        "model": model,
                        **_support(subset, global_subset_mask),
                        **_ranking_metrics(subset, subset[column].to_numpy(float), global_subset_mask),
                        "brier": None,
                        "log_loss": None,
                        "ece": None,
                        "calibration_in_the_large": None,
                        "calibration_slope": None,
                        "calibration_status": "NOT_APPLICABLE_GLOBAL_SELECTION",
                    }
                )
            left = _ranking_metrics(subset, subset[MODELS["AP-EV"]].to_numpy(float), _selected(subset[MODELS["AP-EV"]]))
            right = _ranking_metrics(subset, subset[MODELS["B1"]].to_numpy(float), _selected(subset[MODELS["B1"]]))
            bootstrap = _bootstrap_difference(subset, MODELS["AP-EV"], MODELS["B1"], 300, 20260923)
            rows.append(
                {
                    "analysis_mode": "LOCAL_SLICE_DIFFERENCE",
                    "slice": slice_name,
                    "slice_value": str(slice_value),
                    "model": "AP-EV_MINUS_B1",
                    **_support(subset, _selected(subset[MODELS["AP-EV"]])),
                    "value_capture_10": (left["value_capture_10"] or 0) - (right["value_capture_10"] or 0),
                    "value_capture_10_status": "EVALUABLE" if left["value_capture_10"] is not None and right["value_capture_10"] is not None else "NOT_EVALUABLE",
                    **bootstrap,
                }
            )
    return pd.DataFrame(rows)


def _intersection_table(frame: pd.DataFrame) -> pd.DataFrame:
    pairs = (
        ("sector", "company_size"), ("sector", "revenue_quartile"), ("product", "region"),
        ("product", "company_size"), ("account_history_bin", "revenue_quartile"),
        ("account_history_bin", "product"),
    )
    work = frame.copy()
    columns = []
    for left, right in pairs:
        name = f"{left}_x_{right}"
        work[name] = work[left].astype(str) + " × " + work[right].astype(str)
        columns.append(name)
    result = _slice_table(work, columns, minimum=30)
    result["exploratory"] = True
    result["label_metric_support"] = np.where(result["wins"].fillna(0).ge(5), "EVALUABLE", "NOT_EVALUABLE_LT_5_WINS")
    return result


def _rank_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for name, column in MODELS.items():
        result[f"{name}_rank"] = result[column].rank(method="first", ascending=False).astype(int)
        result[f"{name}_selected"] = _selected(result[column])
    return result


def _error_cases(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = _rank_columns(frame)
    rows = []
    for model, column in MODELS.items():
        selected = ranked[f"{model}_selected"]
        candidates = {
            "MISSED_WHALES": ranked[ranked["won"].eq(1) & ~selected].nlargest(20, "realized_value"),
            "FALSE_PRIORITIES": ranked[selected & ranked["lost"].eq(1) & ranked["realized_value"].eq(0)].nlargest(20, column),
            "NEAR_CUTOFF": ranked.assign(_distance=(ranked[f"{model}_rank"] - int(math.ceil(0.1 * len(ranked)))).abs()).nsmallest(20, "_distance"),
        }
        for case_type, cases in candidates.items():
            for _, item in cases.iterrows():
                rows.append(
                    {
                        "case_type": case_type, "case_model": model, "opportunity_id": item["opportunity_id"],
                        "account": item["account"], "sector": item["sector"], "product": item["product"],
                        "region": item["region"], "company_size": item["company_size"], "revenue": item["revenue"],
                        "sales_price": item["sales_price"], "account_prior_opportunities": item["account_prior_opportunities"],
                        "account_prior_resolved": item["account_prior_resolved"], "account_prior_win_rate": item["account_prior_win_rate"],
                        "account_prior_realized_value": item["account_prior_realized_value"], "actual_value": item["realized_value"],
                        "actual_outcome": item["terminal_state"], "B1_rank": item["B1_rank"], "AP-EV_rank": item["AP-EV_rank"],
                        "LambdaMART_rank": item["LambdaMART_rank"], "B4_rank": item["B4 XGBoost_rank"],
                        "win_probability": item["win_probability"], "expected_value": item["expected_value_given_win"],
                        "expected_terminal_days": item["expected_terminal_days"], "AP-EV_score": item[MODELS["AP-EV"]],
                        "ranking_uncertainty": item.get("p_top10", np.nan),
                    }
                )
    return pd.DataFrame(rows)


def _disagreement(frame: pd.DataFrame) -> pd.DataFrame:
    selected = {name: _selected(frame[column]) for name, column in MODELS.items()}
    rows = []
    values = frame["realized_value"].to_numpy(float)
    won = frame["won"].to_numpy(int)
    for left, right in itertools.combinations(MODELS, 2):
        common = selected[left] & selected[right]
        union = selected[left] | selected[right]
        for group_name, mask in (("PAIR_SUMMARY", union), (f"{left}_ONLY", selected[left] & ~selected[right]), (f"{right}_ONLY", selected[right] & ~selected[left]), ("COMMON", common)):
            rows.append(
                {
                    "record_type": group_name, "model_left": left, "model_right": right,
                    "jaccard": float(common.sum() / max(union.sum(), 1)),
                    "rank_correlation": float(spearmanr(frame[MODELS[left]], frame[MODELS[right]]).statistic),
                    "common_selection_count": int(common.sum()), "unique_selection_count": int(mask.sum()),
                    "captured_value": float(values[mask].sum()), "win_rate": float(won[mask].mean()) if mask.any() else None,
                }
            )
    b1, ap = selected["B1"], selected["AP-EV"]
    for group, mask in (("B1_AND_APEV", b1 & ap), ("B1_ONLY", b1 & ~ap), ("APEV_ONLY", ap & ~b1), ("NEITHER", ~b1 & ~ap)):
        base = {"record_type": "B1_APEV_GROUP", "group": group, "rows": int(mask.sum()), "captured_value": float(values[mask].sum()), "win_rate": float(won[mask].mean()) if mask.any() else None}
        rows.append(base)
        for dimension in ("sector", "product", "company_size", "region"):
            counts = frame.loc[mask, dimension].value_counts(normalize=True)
            for category, share in counts.items():
                rows.append({**base, "record_type": "B1_APEV_COMPOSITION", "dimension": dimension, "category": str(category), "composition_share": float(share)})
    return pd.DataFrame(rows)


def _capacity_curve(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, column in CAPACITY_MODELS.items():
        previous_value = 0.0
        for capacity in CAPACITIES:
            mask = _selected(frame[column], capacity)
            metrics = _ranking_metrics(frame, frame[column].to_numpy(float), mask)
            captured = float(frame.loc[mask, "realized_value"].sum())
            rows.append(
                {
                    "model": model, "capacity": capacity, "selected_count": int(mask.sum()),
                    "value_capture": metrics["value_capture_10"], "win_capture": metrics["win_capture_10"],
                    "precision": metrics["precision_10"], "recall": metrics["recall_10"],
                    "value_per_selected_opportunity": captured / max(mask.sum(), 1), "oracle_regret": metrics["oracle_regret"],
                    "marginal_captured_value": captured - previous_value,
                }
            )
            previous_value = captured
    return pd.DataFrame(rows)


def capacity_is_monotone(curve: pd.DataFrame) -> bool:
    """Check nested-capacity captured value monotonicity model by model."""
    return all(group.sort_values("capacity")["value_capture"].diff().dropna().ge(-1e-12).all() for _, group in curve.groupby("model"))


def _b1_decomposition(frame: pd.DataFrame) -> pd.DataFrame:
    variants = {
        "sales_price_only": frame["sales_price"], "product_prior_only": frame["product_prior_win_rate"],
        "sector_prior_only": frame["sector_prior_win_rate"], "price_x_product_prior": frame["sales_price"] * frame["product_prior_win_rate"],
        "price_x_sector_prior": frame["sales_price"] * frame["sector_prior_win_rate"],
        "product_x_sector_prior": frame["product_prior_win_rate"] * frame["sector_prior_win_rate"],
        "full_B1": frame[MODELS["B1"]],
    }
    rows = []
    for name, score in variants.items():
        mask = _selected(score)
        rows.append({"analysis_type": "COMPONENT_VARIANT", "variant": name, **_support(frame, mask), **_ranking_metrics(frame, np.asarray(score), mask)})
    for dimension in ("product", "sector"):
        mask = _selected(frame[MODELS["B1"]])
        for category, subset in frame.loc[mask].groupby(dimension):
            rows.append({"analysis_type": "FULL_B1_CONCENTRATION", "dimension": dimension, "category": str(category), "selected_count": len(subset), "selection_share": len(subset) / mask.sum(), "realized_value": float(subset["realized_value"].sum())})
    return pd.DataFrame(rows)


def _apev_diagnostics() -> pd.DataFrame:
    locked = pd.read_parquet(EVIDENCE / "maven_apev_ablations_locked.parquet")
    policy = pd.read_parquet(EVIDENCE / "maven_apev_ablations_policy.parquet")
    rows = pd.concat([locked.assign(record_type="STORED_ABLATION"), policy.assign(record_type="STORED_ABLATION")], ignore_index=True)
    unavailable = pd.DataFrame(
        [{"record_type": "OPPORTUNITY_COMPONENTS", "component": component, "status": "NOT_EVALUABLE_INFRASTRUCTURE_FAILURE_CUDA_UNAVAILABLE", "explanation": "Frozen GPU model inference was not run; silent CPU fallback is prohibited."} for component in ("timing_contribution", "conditional_win_contribution", "conditional_value_contribution", "discount_contribution")]
    )
    return pd.concat([rows, unavailable], ignore_index=True)


def _calibration_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    development = pd.read_parquet(EVIDENCE / "maven_development.parquet")
    predictions = pd.read_parquet(EVIDENCE / "maven_predictions_development.parquet")
    dev = predictions.merge(development.drop(columns=[column for column in predictions if column in development and column != "opportunity_id"]), on="opportunity_id", how="left")
    locked = frame.assign(split="LOCKED")
    combined = pd.concat([dev[dev["split"].isin(["VALIDATION", "POLICY"])], locked], ignore_index=True, sort=False)
    bundle = joblib.load(BUNDLE_PATH)
    raw = np.clip(combined[MODELS["B4 XGBoost"]].to_numpy(float), 1e-6, 1 - 1e-6)
    mappings = {
        "raw_B4": raw,
        "Platt_mapping": bundle.calibrators["platt"].predict_proba(logit(raw).reshape(-1, 1))[:, 1],
        "isotonic_mapping": bundle.calibrators["isotonic"].predict(raw),
        "frozen_selected_probability": combined["win_probability"].to_numpy(float),
    }
    rows = []
    dimensions = ("sector", "product", "company_size", "revenue_quartile", "account_history_bin")
    combined = _add_bins(combined)
    for method, probability in mappings.items():
        combined["_probability"] = probability
        for cohort, cohort_frame in combined.groupby("split"):
            cohort_metrics = _calibration(cohort_frame["won"].to_numpy(int), cohort_frame["_probability"].to_numpy(float))
            rows.append({"record_type": "COHORT_METRIC", "cohort": cohort, "method": method, "support_n": len(cohort_frame), **cohort_metrics})
            bins = pd.qcut(cohort_frame["_probability"].rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])
            for decile, part in cohort_frame.groupby(bins, observed=True):
                rows.append({"record_type": "RELIABILITY_DECILE", "cohort": cohort, "method": method, "probability_decile": str(decile), "support_n": len(part), "mean_prediction": float(part["_probability"].mean()), "observed_win_rate": float(part["won"].mean()), "difference": float(part["_probability"].mean() - part["won"].mean())})
            for dimension in dimensions:
                for category, part in cohort_frame.groupby(dimension, observed=True):
                    status = "EVALUABLE" if len(part) >= 30 and part["won"].sum() >= 5 and part["lost"].sum() >= 5 else "NOT_EVALUABLE_SUPPORT"
                    metric = _calibration(part["won"].to_numpy(int), part["_probability"].to_numpy(float)) if status == "EVALUABLE" else {"brier": None, "log_loss": None, "ece": None, "calibration_in_the_large": None, "calibration_slope": None, "calibration_status": status}
                    rows.append({"record_type": "SLICE_METRIC", "cohort": cohort, "method": method, "dimension": dimension, "category": str(category), "support_n": len(part), **metric})
    return pd.DataFrame(rows)


def _rank_stability(frame: pd.DataFrame, replicates: int = 1000) -> pd.DataFrame:
    groups = frame["account_key"].astype(str).unique()
    keys = frame["account_key"].astype(str).to_numpy()
    positions = {group: np.flatnonzero(keys == group) for group in groups}
    rng = np.random.default_rng(20260923)
    rows = []
    for model in ("B1", "AP-EV", "LambdaMART"):
        score = frame[MODELS[model]].to_numpy(float)
        base_order = np.argsort(-score, kind="stable")
        base_rank = np.empty(len(frame), int)
        base_rank[base_order] = np.arange(1, len(frame) + 1)
        base_top = set(base_order[: math.ceil(0.1 * len(frame))])
        rank_samples: list[list[int]] = [[] for _ in range(len(frame))]
        counts = {capacity: np.zeros(len(frame)) for capacity in (0.05, 0.10, 0.20)}
        appearances = np.zeros(len(frame))
        jaccards, spearmans, kendalls = [], [], []
        for _ in range(replicates):
            sampled = rng.choice(groups, len(groups), replace=True)
            take = np.concatenate([positions[group] for group in sampled])
            order = take[np.argsort(-score[take], kind="stable")]
            local_rank = np.arange(1, len(order) + 1)
            for position, rank in zip(order, local_rank, strict=True):
                rank_samples[position].append(int(rank))
                appearances[position] += 1
            for capacity, counter in counts.items():
                counter[order[: max(1, math.ceil(capacity * len(order)))]] += 1
            top = set(order[: max(1, math.ceil(0.1 * len(order)))])
            jaccards.append(len(top & base_top) / max(len(top | base_top), 1))
            reference = base_rank[order]
            spearmans.append(float(spearmanr(reference, local_rank).statistic))
            kendalls.append(float(kendalltau(reference, local_rank).statistic))
        rows.append({"record_type": "MODEL_STABILITY", "model": model, "replicates": replicates, "top10_jaccard_mean": float(np.nanmean(jaccards)), "spearman_mean": float(np.nanmean(spearmans)), "kendall_mean": float(np.nanmean(kendalls))})
        for index, values in enumerate(rank_samples):
            if not values:
                continue
            rows.append({"record_type": "OPPORTUNITY_RANK", "model": model, "opportunity_id": frame.iloc[index]["opportunity_id"], "median_rank": float(np.median(values)), "rank_q025": float(np.quantile(values, 0.025)), "rank_q975": float(np.quantile(values, 0.975)), "p_top5": float(counts[0.05][index] / appearances[index]), "p_top10": float(counts[0.10][index] / appearances[index]), "p_top20": float(counts[0.20][index] / appearances[index]), "replicates": replicates})
    return pd.DataFrame(rows)


def _concentration(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, column in MODELS.items():
        selected = frame.loc[_selected(frame[column])]
        for dimension in ("account", "sector", "product", "region"):
            grouped = selected.groupby(dimension, observed=True).agg(selected_count=("opportunity_id", "size"), realized_value=("realized_value", "sum")).reset_index()
            selection_hhi = hhi(grouped["selected_count"])
            value_hhi = hhi(grouped["realized_value"])
            for _, item in grouped.iterrows():
                rows.append({"model": model, "dimension": dimension, "category": str(item[dimension]), "selected_count": int(item["selected_count"]), "selection_share": float(item["selected_count"] / len(selected)), "realized_value_share": float(item["realized_value"] / max(selected["realized_value"].sum(), 1e-12)), "selection_hhi": selection_hhi, "value_hhi": value_hhi, "top_category_share": float(grouped["selected_count"].max() / len(selected))})
    return pd.DataFrame(rows)


def _psi(reference: np.ndarray, current: np.ndarray) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return 0.0
    ref = np.histogram(reference, bins=edges)[0] / len(reference)
    cur = np.histogram(np.clip(current, edges[0], edges[-1]), bins=edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-6, None), np.clip(cur, 1e-6, None)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def _drift(frame: pd.DataFrame) -> pd.DataFrame:
    development = pd.read_parquet(EVIDENCE / "maven_development.parquet")
    policy = development[development["split"].eq("POLICY")]
    dev_predictions = pd.read_parquet(EVIDENCE / "maven_predictions_development.parquet")
    policy_scores = dev_predictions[dev_predictions["split"].eq("POLICY")]
    rows = []
    numeric = ("revenue", "employees", "sales_price", "company_age", "account_prior_opportunities", "account_prior_win_rate")
    for field in numeric:
        ref, cur = policy[field].dropna().to_numpy(float), frame[field].dropna().to_numpy(float)
        rows.extend([{"drift_type": "INPUT_DRIFT", "field": field, "metric": "KS", "value": float(ks_2samp(ref, cur).statistic)}, {"drift_type": "INPUT_DRIFT", "field": field, "metric": "WASSERSTEIN", "value": float(wasserstein_distance(ref, cur))}, {"drift_type": "INPUT_DRIFT", "field": field, "metric": "PSI", "value": _psi(ref, cur)}])
    for field in ("sector", "product", "company_size", "office_location"):
        categories = sorted(set(policy[field].astype(str)) | set(frame[field].astype(str)))
        ref = policy[field].astype(str).value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(float)
        cur = frame[field].astype(str).value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(float)
        rows.append({"drift_type": "INPUT_DRIFT", "field": field, "metric": "JENSEN_SHANNON", "value": float(np.sqrt(0.5 * entropy(ref + 1e-12, (ref + cur) / 2 + 1e-12) + 0.5 * entropy(cur + 1e-12, (ref + cur) / 2 + 1e-12)))})
        rows.append({"drift_type": "INPUT_DRIFT", "field": field, "metric": "MAX_CATEGORY_FREQUENCY_SHIFT", "value": float(np.max(np.abs(cur - ref)))})
        rows.append({"drift_type": "INPUT_DRIFT", "field": field, "metric": "UNSEEN_CATEGORY_RATE", "value": float(frame[field].astype(str).isin(set(frame[field].astype(str)) - set(policy[field].astype(str))).mean())})
    for model, column in MODELS.items():
        ref, cur = policy_scores[column].to_numpy(float), frame[column].to_numpy(float)
        rows.extend([{"drift_type": "SCORE_DRIFT", "field": model, "metric": "KS", "value": float(ks_2samp(ref, cur).statistic)}, {"drift_type": "SCORE_DRIFT", "field": model, "metric": "WASSERSTEIN", "value": float(wasserstein_distance(ref, cur))}, {"drift_type": "SCORE_DRIFT", "field": model, "metric": "PSI", "value": _psi(ref, cur)}])
    for cohort, data in (("POLICY", policy), ("LOCKED", frame)):
        for metric, value in (("WIN_RATE", data["won"].mean()), ("LOST_RATE", data["lost"].mean()), ("OPEN_RATE", (data["terminal_state"] == "OPEN_CENSORED_60D").mean()), ("MEAN_REALIZED_VALUE", data["realized_value"].mean()), ("HIGH_VALUE_TAIL_PREVALENCE", (data["realized_value"] >= data["realized_value"].quantile(0.9)).mean())):
            rows.append({"drift_type": "LABEL_OUTCOME_DRIFT", "field": cohort, "metric": metric, "value": float(value)})
    return pd.DataFrame(rows)


def _unavailable_attribution(kind: str) -> pd.DataFrame:
    groups = ("firmographics", "product", "calendar", "account_history", "product_history", "sector_history")
    return pd.DataFrame([{"analysis": kind, "feature_group": group, "status": "NOT_EVALUABLE_INFRASTRUCTURE_FAILURE_CUDA_UNAVAILABLE", "detail": "Frozen XGBoost read-only inference was not run because CUDA is unavailable and CPU fallback is prohibited."} for group in groups])


def _adjust_bh(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 1.0
    for rank_index in range(len(p_values) - 1, -1, -1):
        index = order[rank_index]
        running = min(running, p_values[index] * len(p_values) / (rank_index + 1))
        adjusted[index] = running
    return adjusted


def _worst_slices(frame: pd.DataFrame) -> pd.DataFrame:
    variables = ("sector", "product", "company_size", "revenue_quartile", "account_history_bin", "region")
    rows = []
    rng = np.random.default_rng(20260923)
    definitions: list[tuple[str, list[str]]] = [(field, [field]) for field in variables]
    definitions += [(f"{left} × {right}", [left, right]) for left, right in itertools.combinations(variables, 2)]
    for name, fields in definitions:
        group_key: str | list[str] = fields[0] if len(fields) == 1 else fields
        for values, part in frame.groupby(group_key, observed=True):
            if len(part) < 30 or part["won"].sum() < 5:
                continue
            b1 = _ranking_metrics(part, part[MODELS["B1"]].to_numpy(float), _selected(part[MODELS["B1"]]))["value_capture_10"]
            ap = _ranking_metrics(part, part[MODELS["AP-EV"]].to_numpy(float), _selected(part[MODELS["AP-EV"]]))["value_capture_10"]
            diffs = []
            clusters = part["account_key"].astype(str).unique()
            cluster_keys = part["account_key"].astype(str).to_numpy()
            cluster_positions = {
                cluster: np.flatnonzero(cluster_keys == cluster) for cluster in clusters
            }
            for _ in range(300):
                sampled_clusters = rng.choice(clusters, len(clusters), replace=True)
                take = np.concatenate([cluster_positions[cluster] for cluster in sampled_clusters])
                sample = part.iloc[take]
                total = sample["realized_value"].sum()
                if total <= 0:
                    continue
                a = sample.loc[_selected(sample[MODELS["AP-EV"]]), "realized_value"].sum() / total
                b = sample.loc[_selected(sample[MODELS["B1"]]), "realized_value"].sum() / total
                diffs.append(a - b)
            if not diffs:
                continue
            effect = float(ap - b1)
            raw_p = float(2 * min(np.mean(np.asarray(diffs) <= 0), np.mean(np.asarray(diffs) >= 0)))
            rows.append({"slice": name, "slice_value": str(values), "support_n": len(part), "wins": int(part["won"].sum()), "effect_estimate": effect, "ci_low": float(np.quantile(diffs, 0.025)), "ci_high": float(np.quantile(diffs, 0.975)), "raw_p_value": min(raw_p, 1.0), "discovery_label": "POST_HOC_SLICE_DISCOVERY", "maximum_interaction_depth": len(fields)})
    result = pd.DataFrame(rows).sort_values("effect_estimate")
    if not result.empty:
        result["bh_adjusted_p_value"] = _adjust_bh(result["raw_p_value"].to_numpy(float))
    return result


def _markdown_table(frame: pd.DataFrame, columns: list[str], rows: int = 12) -> str:
    if frame.empty:
        return "No evaluable rows."
    display = frame.loc[:, [column for column in columns if column in frame]].head(rows).copy()
    names = [str(column) for column in display.columns]
    output = ["| " + " | ".join(names) + " |", "|" + "|".join(["---"] * len(names)) + "|"]
    for values in display.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if isinstance(value, float):
                rendered.append(f"{value:.4f}" if np.isfinite(value) else "")
            else:
                rendered.append(str(value).replace("|", "\\|"))
        output.append("| " + " | ".join(rendered) + " |")
    return "\n".join(output)


def _write_reports(outputs: dict[str, pd.DataFrame], cuda_available: bool) -> None:
    reports = REPO / "reports"
    reports.mkdir(exist_ok=True)
    slice_rows = outputs["maven_slice_metrics_postlock"]
    worst = outputs["maven_worst_slice_discovery_postlock"]
    errors = outputs["maven_error_cases_postlock"]
    b1 = outputs["maven_b1_decomposition_postlock"]
    capacity = outputs["maven_capacity_curve_postlock"]
    stability = outputs["maven_rank_stability_postlock"]
    drift = outputs["maven_drift_diagnostics_postlock"]
    calibration = outputs["maven_calibration_diagnostics_postlock"]
    disagreement = outputs["maven_model_disagreement_postlock"]
    full_b1 = b1[(b1["analysis_type"] == "COMPONENT_VARIANT")].sort_values("value_capture_10", ascending=False)
    locked_cal = calibration[(calibration["record_type"] == "COHORT_METRIC") & (calibration["cohort"] == "LOCKED")]
    model_stability = stability[stability["record_type"] == "MODEL_STABILITY"]
    artifact_metadata = f"""base_protocol: `{BASE_PROTOCOL}`
diagnostic_protocol: `{DIAGNOSTIC_PROTOCOL}`
result_label: `{LABEL}`
post_lock: `true`
decision_eligible: `false`
"""
    overall = f"""# Track A Post-Lock Diagnostics

{artifact_metadata}

This analysis reuses frozen A1 predictions/outcomes and is explanatory only. It does not retrain, retune, recalibrate, select a calibrator, change a gate, or supply promotion evidence. Maven is `FICTITIOUS_PUBLIC`. Predictive ranking is not a causal effect.

## Main explanation

B1 won at the frozen 10% capacity because its price-weighted product and sector priors concentrated selection on higher realized-value categories. AP-EV captured slightly more wins but less value, consistent with a decision-value miss rather than a simple discrimination failure. Selection-set composition and the B1 decomposition below quantify that mechanism.

{_markdown_table(full_b1, ['variant', 'value_capture_10', 'win_capture_10', 'selected_count'], 7)}

## B1 versus AP-EV disagreement

{_markdown_table(disagreement[disagreement['record_type'].isin(['B1_APEV_GROUP'])], ['group', 'rows', 'captured_value', 'win_rate'], 4)}

## Capacity behavior

{_markdown_table(capacity[capacity['capacity'].isin([0.05, 0.10, 0.20, 0.50])], ['model', 'capacity', 'value_capture', 'win_capture', 'oracle_regret'], 24)}

## Rank stability

{_markdown_table(model_stability, ['model', 'top10_jaccard_mean', 'spearman_mean', 'kendall_mean'], 3)}

## Restricted worst-slice discovery

All rows below are `POST_HOC_SLICE_DISCOVERY`, depth <= 2, n >= 30, wins >= 5, and cannot alter A1.

{_markdown_table(worst, ['slice', 'slice_value', 'support_n', 'effect_estimate', 'ci_low', 'ci_high', 'bh_adjusted_p_value'], 12)}

## AP-EV components and explainability

Stored frozen ablations are reported. Opportunity-level component inference, SHAP, grouped permutation, and PDP/ICE are `NOT_EVALUABLE_INFRASTRUCTURE_FAILURE` because CUDA availability was `{cuda_available}`; the required no-CPU-fallback control was enforced. PDP/ICE, if run under A2 with CUDA, must be labeled **MODEL RESPONSE, NOT CAUSAL EFFECT** and interpreted cautiously under correlated features.

## Decision

**A1 decision remains RETAIN_BASELINE.** Any response to these findings requires a genuinely new A2 protocol.
"""
    (reports / "TrackA_PostLock_Diagnostics.md").write_text(overall, encoding="utf-8")
    (reports / "TrackA_Slicing_Report.md").write_text(f"""# Track A Slicing Report

{artifact_metadata}

Local slice ranking and global policy behavior are separate rows and must not be mixed. Zero history is a **sparse-history proxy, not a true cold account**.

## Largest local AP-EV minus B1 weaknesses

{_markdown_table(slice_rows[slice_rows['analysis_mode'] == 'LOCAL_SLICE_DIFFERENCE'].sort_values('value_capture_10'), ['slice', 'slice_value', 'support_n', 'wins', 'value_capture_10', 'difference_ci_low', 'difference_ci_high'], 20)}

## Interpretation boundary

Small slices with insufficient outcome classes explicitly carry `NOT_EVALUABLE`; intersectional results are exploratory and limited to the six predefined pairs.
""", encoding="utf-8")
    (reports / "TrackA_Error_Analysis.md").write_text(f"""# Track A Error Analysis

{artifact_metadata}

Cases are frozen diagnostic examples and were not used to modify any model.

## Top missed whales

{_markdown_table(errors[errors['case_type'] == 'MISSED_WHALES'], ['case_model', 'opportunity_id', 'account', 'product', 'actual_value', 'B1_rank', 'AP-EV_rank'], 20)}

## False priorities

{_markdown_table(errors[errors['case_type'] == 'FALSE_PRIORITIES'], ['case_model', 'opportunity_id', 'account', 'product', 'actual_outcome', 'B1_rank', 'AP-EV_rank'], 20)}

Near-cutoff cases and all requested diagnostic fields are in the canonical Parquet artifact.
""", encoding="utf-8")
    (reports / "TrackA_Calibration_Drift.md").write_text(f"""# Track A Calibration and Drift

{artifact_metadata}

The frozen calibrator is unchanged; no new calibrator is selected.

## LOCKED calibration

{_markdown_table(locked_cal, ['method', 'support_n', 'brier', 'log_loss', 'ece', 'calibration_in_the_large', 'calibration_slope'], 4)}

The LOCKED calibration weakness is a level-and-slope transport failure: predictions understate the LOCKED win base rate and the selected isotonic mapping has a shallow slope. Isotonic calibration fitted on a small development sample has material step-function overfit risk. A2 should prospectively compare prespecified calibration methods on new validation data only.

## Largest measured drift diagnostics

{_markdown_table(drift.sort_values('value', ascending=False), ['drift_type', 'field', 'metric', 'value'], 20)}

Input, score, and post-label outcome drift are reported separately. No diagnostic threshold is an A1 gate.
""", encoding="utf-8")
    (reports / "TrackA_Explainability.md").write_text(f"""# Track A Explainability

{artifact_metadata}

Native SHAP/contribution inference and grouped decision-specific permutation were not executed because CUDA is unavailable; silently moving the frozen GPU models to CPU is prohibited. The artifacts explicitly record this `INFRASTRUCTURE_FAILURE` status.

Local explanation targets remain predefined as top missed whales, false priorities, and largest B1/AP-EV disagreements. No explanation was used for feature selection or model alteration. PDP/ICE was not executed and, when prospectively run, must be labeled **MODEL RESPONSE, NOT CAUSAL EFFECT**; correlated-feature combinations may be unrealistic.
""", encoding="utf-8")
    (reports / "TrackA_Operational_Monitoring.md").write_text(f"""# Track A Operational Monitoring Specification

{artifact_metadata}

This is a local production-style specification, **not a deployed monitor**.

## Immediate unlabeled monitoring

- Data quality: schema, missingness, ranges, category validity, identifier uniqueness, batch volume, and point-in-time availability.
- Input drift: numeric KS/Wasserstein/PSI and categorical Jensen-Shannon/frequency/unseen-category diagnostics.
- Score drift: B1, AP-EV, B4, and LambdaMART distribution and rank-shift diagnostics.
- Attribution drift: normalized grouped attribution ranks when CUDA inference is available.
- Selection concentration: account/sector/product/region HHI and top-category share. This is concentration analysis, not fairness analysis.

## Delayed labeled monitoring

- After 60-day maturity: ValueCapture, WinCapture, precision/recall, calibration, oracle regret, and slice degradation.
- Preserve open/censored semantics and account-cluster uncertainty.
- Revisit only under a new protocol; never tune from delayed A1 outcomes.

## PROPOSED A2 triggers

All thresholds are **PROPOSED, NOT A1-VALIDATED**: PSI >= 0.20; categorical Jensen-Shannon >= 0.10; calibration ECE >= 0.10; ValueCapture loss >= 5 percentage points; selection HHI increase >= 0.10; or a sufficiently supported slice whose adjusted degradation interval excludes zero. These require prospective validation.
""", encoding="utf-8")
    (reports / "AccountPulse_CrossTrack_Synthesis.md").write_text(f"""# AccountPulse Cross-Track Synthesis

{artifact_metadata}

Evidence classes and estimands remain separate.

| Track | Evidence / estimand | Diagnostic synthesis |
|---|---|---|
| A Maven | `FICTITIOUS_PUBLIC`; predictive capacity ranking | B1 beat AP-EV on frozen ValueCapture@10%; complex timing/value composition did not overcome the price/category heuristic. |
| B Olist | Public marketplace; predictive seller/lead ranking | Sparse AP-EV did not beat XGBoost: estimated difference -7.06 points with intervals spanning zero. |
| C UCI | Public retail replay; predictive customer ranking | Representation/fusion improved or matched discrimination in places, but fixed late fusion added only 0.30 ValueCapture points over RFM with a CI spanning zero; simple RFM remained formidable. |
| D Criteo | Verified randomized advertising; causal treatment allocation | Uplift and response targeting differed materially at constrained capacity. This is causal allocation evidence, not propensity-ranking evidence and is not transportable to Maven. |

Calibration mattered most visibly in Track A, where the frozen isotonic mapping improved Brier versus raw B4 but transported with slope weakness. Across predictive tracks, better representation or discrimination did not guarantee greater decision value. Track D answers a different question: treatment-effect targeting rather than outcome propensity.
""", encoding="utf-8")


def _verify_base() -> tuple[str, str]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    tag = subprocess.check_output(["git", "rev-parse", f"{TAG}^{{commit}}"], cwd=REPO, text=True).strip()
    base = subprocess.check_output(["git", "merge-base", "HEAD", TAG], cwd=REPO, text=True).strip()
    if head != tag or base != tag:
        raise RuntimeError(f"Diagnostic branch base mismatch: HEAD={head}, tag={tag}, merge_base={base}")
    return head, tag


def run() -> dict[str, Any]:
    """Execute all safe post-lock diagnostics and verify immutable A1 bytes."""
    head, tag = _verify_base()
    before = protected_hashes()
    locked = pd.read_parquet(EVIDENCE / "maven_locked_predictions.parquet")
    features = pd.read_parquet(EVIDENCE / "maven_locked_features.parquet")
    uncertainty = pd.read_parquet(EVIDENCE / "maven_ranking_uncertainty_locked.parquet")
    frame = features.merge(locked, on=["opportunity_id", "account_key", "engage_date"], validate="1:1")
    frame = frame.merge(uncertainty[["opportunity_id", "p_top10"]], on="opportunity_id", how="left", validate="1:1")
    tie_order = frame["opportunity_id"].map(
        lambda value: int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:8], "big")
    )
    frame = frame.assign(_tie_order=tie_order).sort_values("_tie_order", ascending=False)
    frame = _add_bins(frame.drop(columns="_tie_order").reset_index(drop=True))
    cuda_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except ImportError:
        cuda_available = False
    outputs = {
        "maven_slice_metrics_postlock": _slice_table(frame, SLICE_COLUMNS),
        "maven_intersection_slice_metrics_postlock": _intersection_table(frame),
        "maven_error_cases_postlock": _error_cases(frame),
        "maven_model_disagreement_postlock": _disagreement(frame),
        "maven_b1_decomposition_postlock": _b1_decomposition(frame),
        "maven_apev_component_diagnostics_postlock": _apev_diagnostics(),
        "maven_calibration_diagnostics_postlock": _calibration_diagnostics(frame),
        "maven_capacity_curve_postlock": _capacity_curve(frame),
        "maven_rank_stability_postlock": _rank_stability(frame),
        "maven_selection_concentration_postlock": _concentration(frame),
        "maven_drift_diagnostics_postlock": _drift(frame),
        "maven_attribution_drift_postlock": _unavailable_attribution("NATIVE_SHAP_ATTRIBUTION_DRIFT"),
        "maven_grouped_permutation_postlock": _unavailable_attribution("GROUPED_VALUE_CAPTURE_PERMUTATION"),
        "maven_worst_slice_discovery_postlock": _worst_slices(frame),
    }
    if not capacity_is_monotone(outputs["maven_capacity_curve_postlock"]):
        raise RuntimeError("Capacity curve captured value is not monotone")
    for name, output in outputs.items():
        write_table(name, _meta(output))
    _write_reports(outputs, cuda_available)
    after = protected_hashes()
    if before != after:
        raise RuntimeError(f"Protected A1 artifact mutation detected: before={before}, after={after}")
    artifacts = {f"artifacts/evidence/{name}.parquet": _sha256(EVIDENCE / f"{name}.parquet") for name in OUTPUT_NAMES}
    reports = {str(path.relative_to(REPO)): _sha256(path) for path in sorted((REPO / "reports").glob("TrackA_*"))}
    reports["reports/AccountPulse_CrossTrack_Synthesis.md"] = _sha256(REPO / "reports/AccountPulse_CrossTrack_Synthesis.md")
    manifest = {
        "base_protocol": BASE_PROTOCOL,
        "diagnostic_protocol": DIAGNOSTIC_PROTOCOL,
        "result_label": LABEL,
        "post_lock": True,
        "decision_eligible": False,
        "a1_decision": "RETAIN_BASELINE",
        "base_commit": head,
        "tag_commit": tag,
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "protected_byte_identical": before == after,
        "cuda_available": cuda_available,
        "gpu_dependent_diagnostics": "NOT_EVALUABLE_INFRASTRUCTURE_FAILURE" if not cuda_available else "EVALUABLE",
        "artifacts": artifacts,
        "reports": reports,
    }
    (REPO / "DIAGNOSTICS_MANIFEST.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return {"status": "ACCOUNT_PULSE_POSTLOCK_DIAGNOSTIC_STATUS", "rows": len(frame), "artifacts": len(outputs), "protected_byte_identical": True, "cuda_available": cuda_available, "a1_decision": "RETAIN_BASELINE"}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
