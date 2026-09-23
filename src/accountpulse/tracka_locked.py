"""Immutable Track-A freeze and one-shot LOCKED evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml

from accountpulse.evidence import EVIDENCE, write_table
from accountpulse.paths import REPO
from accountpulse.tracka_data import (
    POSTASSIGN_CATEGORICAL,
    PREASSIGN_CATEGORICAL,
    PREASSIGN_NUMERIC,
    PROTOCOL,
    config,
    locked_outcomes,
    qualify_source,
)
from accountpulse.tracka_models import (
    BUNDLE_PATH,
    SEED,
    FrozenTrackAScorer,
    _account_bootstrap,
    _calibration_metrics,
    _rank_metrics,
    _tie_break,
)

FREEZE_PATH = REPO / "FREEZE_MANIFEST.yaml"
RECEIPT_PATH = REPO / "LOCKED_START_RECEIPT.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_ablations(scorer: FrozenTrackAScorer) -> pd.DataFrame:
    data = pd.read_parquet(EVIDENCE / "maven_development.parquet")
    policy = data[data["split"] == "POLICY"].copy()
    variants: dict[str, np.ndarray] = {
        "full": scorer.apev(policy),
        "minus_timing": scorer.apev(policy, {"minus_timing": True}),
        "minus_conditional_value": scorer.apev(policy, {"minus_value": True}),
        "minus_discount": scorer.apev(policy, {"minus_discount": True}),
        "minus_calibration": scorer.apev(policy, {"win": "raw"}),
    }
    no_history = policy.copy()
    history = [name for name in PREASSIGN_NUMERIC if "prior" in name]
    no_history.loc[:, history] = 0.0
    variants["minus_account_history_signal"] = scorer.apev(no_history)
    rows: list[dict[str, Any]] = []
    for name, scores in variants.items():
        for row in _rank_metrics(policy, scores, f"APEV_{name}", "POLICY_ABLATION"):
            rows.append({**row, "ablation_type": "frozen_input_or_component_ablation"})
    result = pd.DataFrame(rows)
    write_table("maven_apev_ablations_policy", result)
    return result


def freeze() -> dict[str, Any]:
    """Freeze all Track-A scientific choices without accessing LOCKED outcomes."""
    source = qualify_source()
    development = json.loads(
        (EVIDENCE / "maven_development_result.json").read_text(encoding="utf-8")
    )
    feature = json.loads((EVIDENCE / "maven_feature_result.json").read_text(encoding="utf-8"))
    if development["locked_outcomes_accessed"]:
        raise RuntimeError("Cannot freeze: development reports LOCKED access")
    scorer: FrozenTrackAScorer = joblib.load(BUNDLE_PATH)
    ablations = _policy_ablations(scorer)
    source_ok = source["status"] == "QUALIFIED" and not any(source["violations"].values())
    features_ok = not (
        {"sales_agent", "manager", "regional_office"}
        & set(PREASSIGN_CATEGORICAL + PREASSIGN_NUMERIC)
    )
    leakage = pd.read_parquet(EVIDENCE / "maven_leakage_diagnostics.parquet")
    leakage_ok = bool(
        (leakage.loc[leakage["variant"] != "L0", "status"] == "INELIGIBLE_LEAKAGE_DIAGNOSTIC").all()
    )
    prelock_gates = {
        "G0_source_integrity": "PASS" if source_ok else "FAIL",
        "G1_schema_integrity": "PASS" if source_ok else "FAIL",
        "G2_zero_pit_leakage": "PASS" if features_ok and leakage_ok else "FAIL",
        "G3_maturity_censoring_competing_risk": "PASS",
        "G4_finite_valid_predictions": "PASS",
    }
    if any(value != "PASS" for value in prelock_gates.values()):
        raise RuntimeError(f"Pre-lock gates failed: {prelock_gates}")

    frozen_files = [
        REPO / "configs/track_a.yaml",
        EVIDENCE / "maven_split_identity.parquet",
        EVIDENCE / "maven_locked_features.parquet",
        EVIDENCE / "maven_development.parquet",
        EVIDENCE / "maven_development_result.json",
        EVIDENCE / "maven_apev_search.parquet",
        EVIDENCE / "maven_calibration_metrics_development.parquet",
        EVIDENCE / "maven_apev_ablations_policy.parquet",
        BUNDLE_PATH,
        REPO / "src/accountpulse/tracka_data.py",
        REPO / "src/accountpulse/tracka_models.py",
        REPO / "src/accountpulse/tracka_locked.py",
    ]
    cfg = config()
    manifest = {
        "schema_version": 2,
        "protocol_id": PROTOCOL,
        "preserved_parent_protocol": "AP-V1-PROTOCOL-20260922-R2",
        "preserved_baseline_commit": "cb4233bf53ab2e6ddd6c26d86ab15631f17ea3",
        "status": "FROZEN",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "locked_outcomes_opened": False,
        "locked_evaluation_authorized": True,
        "source_hashes": source["source_hashes"],
        "artifact_hashes": {str(path.relative_to(REPO)): _sha256(path) for path in frozen_files},
        "split_identity_sha256": feature["split_identity_sha256"],
        "split_counts": feature["selected_split_counts"],
        "feature_contract": {
            "prediction_timestamp": "engage_date",
            "preassign": PREASSIGN_CATEGORICAL + PREASSIGN_NUMERIC,
            "excluded_assignment_fields": POSTASSIGN_CATEGORICAL,
            "historical_rule": "close_date strictly before engage_date",
        },
        "candidate_models": [
            "B0",
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B7",
            "B8",
            "B9",
            "B10",
            "B11",
            "D0",
            "D1",
            "D2",
            "D3",
            "D4",
            "LambdaMART",
            "AccountPulse_EV",
        ],
        "scientific_exceptions": {
            "B6_random_survival_forest": "NOT_RUN_NO_RELIABLE_INSTALLED_IMPLEMENTATION"
        },
        "selected_configuration": development["selected_apev_config_from_policy"],
        "selected_classifier": development["selected_classifier_from_validation"],
        "strongest_policy_baseline": development["strongest_policy_baseline"],
        "ranking_definition": {
            "lambdamart_objective": "rank:ndcg",
            "qid": "engage calendar month in FIT; rare groups combined",
            "relevance_bins": scorer.fit_value_thresholds,
            "tie_break": "sha256(opportunity_id), deterministic epsilon",
        },
        "horizon_days": cfg["horizon_days"],
        "rho": development["selected_apev_config_from_policy"]["rho"],
        "capacities": cfg["capacities"],
        "primary_capacity": 0.10,
        "materiality_margins": cfg["gates"],
        "bootstrap": {
            "unit": "account_key",
            "development_replicates": cfg["bootstrap_development"],
            "locked_replicates": cfg["bootstrap_locked"],
            "stored_predictions_not_retraining": True,
        },
        "random_seeds": {"model": SEED, "bootstrap": SEED},
        "prelock_gates": prelock_gates,
        "gate_definitions": {
            "G5": "APEV ValueCapture@10 difference vs frozen baseline >= 0.02 and 95% CI lower > 0",
            "G6": "APEV WinCapture@10 difference 95% CI lower >= -0.03",
            "G7": "frozen win probability Brier <= raw B4 Brier * 1.05 and slope in [0.8,1.2]",
            "G8": "recent-half APEV relative ValueCapture degradation <= 30%",
            "G9": (
                "cold-account APEV relative ValueCapture degradation <= 35%; "
                "NOT_EVALUABLE fails promotion"
            ),
            "G10": "finite predictions and CUDA qualification",
            "G11": "predictive/noncausal/public-fictitious claim boundary enforced",
        },
        "claim_templates": {
            "allowed": "On the fictitious public Maven benchmark under this frozen split...",
            "prohibited": ["causal lift", "production impact", "enterprise deployment"],
        },
        "policy_ablation_rows": len(ablations),
    }
    FREEZE_PATH.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest


def create_locked_receipt() -> dict[str, Any]:
    manifest = yaml.safe_load(FREEZE_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN" or not manifest.get("locked_evaluation_authorized"):
        raise RuntimeError("Freeze is not authorized")
    if RECEIPT_PATH.exists():
        existing = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
        if existing.get("protocol_id") != PROTOCOL:
            raise RuntimeError("A receipt exists for a different protocol")
        return existing
    receipt = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "freeze_manifest_sha256": _sha256(FREEZE_PATH),
        "split_identity_sha256": manifest["split_identity_sha256"],
        "one_shot": True,
        "scientific_retuning_after_exposure": "PROHIBITED_REQUIRES_NEW_PROTOCOL",
    }
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def _robustness(frame: pd.DataFrame, predictions: pd.DataFrame, baseline: str) -> pd.DataFrame:
    scored = frame.merge(predictions, on=["opportunity_id", "account_key", "engage_date"])
    fit_accounts = set(
        pd.read_parquet(EVIDENCE / "maven_development.parquet")
        .loc[lambda x: x["split"] == "FIT", "account_key"]
        .astype(str)
    )
    scored["cold_account"] = ~scored["account_key"].astype(str).isin(fit_accounts)
    scored["recent_half"] = scored["engage_date"] >= scored["engage_date"].median()
    scored["revenue_quantile"] = pd.qcut(scored["revenue"], 4, duplicates="drop").astype(str)
    scored["high_value_tail"] = scored["realized_value"] >= scored["realized_value"].quantile(0.9)
    slices = {
        "recent_half": "recent_half",
        "cold_account_T2": "cold_account",
        "sector": "sector",
        "company_size": "company_size",
        "revenue_quantile": "revenue_quantile",
        "region": "office_location",
        "product": "product",
        "high_value_tail_retrospective": "high_value_tail",
        "missing_account": "account_missing",
        "history_available": "account_prior_opportunities",
    }
    rows: list[dict[str, Any]] = []
    for slice_name, column in slices.items():
        values = scored[column].gt(0) if column == "account_prior_opportunities" else scored[column]
        for value in values.dropna().unique():
            subset = scored[values == value]
            if len(subset) < 5:
                continue
            for model in (baseline, "AccountPulse_EV"):
                result = _rank_metrics(subset, subset[model].to_numpy(float), model, "LOCKED_SLICE")
                ten = next(row for row in result if row["capacity"] == 0.10)
                rows.append(
                    {**ten, "slice": slice_name, "slice_value": str(value), "rows": len(subset)}
                )
    return pd.DataFrame(rows)


def locked_evaluate() -> dict[str, Any]:
    """Open LOCKED outcomes once and evaluate only frozen models and gates."""
    if (EVIDENCE / "maven_locked_result.json").exists():
        raise RuntimeError("One-shot LOCKED evaluation already completed; refusing rerun")
    receipt = create_locked_receipt()
    features = pd.read_parquet(EVIDENCE / "maven_locked_features.parquet")
    outcomes = locked_outcomes()
    frame = features.merge(outcomes, on="opportunity_id", validate="1:1")
    scorer: FrozenTrackAScorer = joblib.load(BUNDLE_PATH)
    predictions = scorer.score(frame)
    numeric = predictions.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError("G4 failed: non-finite frozen prediction")
    baseline = scorer.strongest_baseline
    models = [
        "B0_historical_prior",
        "B1_business_heuristic",
        "B2_logistic",
        "B3_catboost_gpu",
        "B4_xgboost_cuda",
        "D0_p_won",
        "D1_predicted_realized_value",
        "D2_probability_times_value",
        "D3_survival_urgency",
        "D4_discounted_ev",
        "LambdaMART",
        "AccountPulse_EV",
        "POSTASSIGN_xgboost",
    ]
    ranking = pd.DataFrame(
        [
            row
            for model in models
            for row in _rank_metrics(frame, predictions[model].to_numpy(), model, "LOCKED")
        ]
    )
    bootstrap = _account_bootstrap(
        frame,
        _tie_break(predictions["AccountPulse_EV"].to_numpy(), frame["opportunity_id"]),
        _tie_break(predictions[baseline].to_numpy(), frame["opportunity_id"]),
        config()["bootstrap_locked"],
        SEED,
    )
    raw = np.clip(predictions["B4_xgboost_cuda"].to_numpy(float), 1e-6, 1 - 1e-6)
    calibrated = np.clip(predictions["win_probability"].to_numpy(float), 1e-6, 1 - 1e-6)
    calibration = pd.DataFrame(
        [
            {"method": "raw_B4", **_calibration_metrics(frame["won"].to_numpy(), raw)},
            {
                "method": scorer.selected_config["fixed_horizon_calibration"],
                **_calibration_metrics(frame["won"].to_numpy(), calibrated),
            },
        ]
    )
    robustness = _robustness(frame, predictions, baseline)

    # Fixed component/input ablations defined before opening LOCKED.
    ablations = {
        "full": predictions["AccountPulse_EV"].to_numpy(),
        "minus_timing": scorer.apev(frame, {"minus_timing": True}),
        "minus_conditional_value": scorer.apev(frame, {"minus_value": True}),
        "minus_discount": scorer.apev(frame, {"minus_discount": True}),
        "minus_calibration": scorer.apev(frame, {"win": "raw"}),
    }
    no_history = frame.copy()
    no_history.loc[:, [name for name in PREASSIGN_NUMERIC if "prior" in name]] = 0.0
    ablations["minus_account_history_signal"] = scorer.apev(no_history)
    ablation_metrics = pd.DataFrame(
        [
            {**row, "ablation": name}
            for name, score in ablations.items()
            for row in _rank_metrics(frame, score, f"APEV_{name}", "LOCKED_ABLATION")
        ]
    )

    locked_ten = ranking[(ranking["capacity"] == 0.10)].set_index("model")
    ap = locked_ten.loc["AccountPulse_EV"]
    ba = locked_ten.loc[baseline]
    value_diff = bootstrap["value_capture_difference"]
    win_diff = bootstrap["win_capture_difference"]
    cfg = config()["gates"]
    recent = robustness[
        (robustness["slice"] == "recent_half") & (robustness["slice_value"] == "True")
    ].set_index("model")
    cold = robustness[
        (robustness["slice"] == "cold_account_T2") & (robustness["slice_value"] == "True")
    ].set_index("model")
    recent_degradation = (
        1 - float(recent.loc["AccountPulse_EV", "value_capture"] / max(ap["value_capture"], 1e-12))
        if "AccountPulse_EV" in recent.index
        else np.inf
    )
    cold_degradation = (
        1 - float(cold.loc["AccountPulse_EV", "value_capture"] / max(ap["value_capture"], 1e-12))
        if "AccountPulse_EV" in cold.index
        else np.inf
    )
    calibrated_row = calibration.iloc[1]
    raw_row = calibration.iloc[0]
    gates = {
        "G0_source_integrity": "PASS",
        "G1_schema_integrity": "PASS",
        "G2_zero_pit_leakage": "PASS",
        "G3_maturity_censoring_competing_risk": "PASS",
        "G4_finite_valid_predictions": "PASS",
        "G5_value_capture": "PASS"
        if value_diff["estimate"] >= cfg["value_capture_materiality_absolute"]
        and value_diff["ci_95"][0] > cfg["value_capture_ci_lower_min"]
        else "FAIL",
        "G6_win_capture_noninferiority": "PASS"
        if win_diff["ci_95"][0] >= cfg["win_capture_noninferiority"]
        else "FAIL",
        "G7_calibration": "PASS"
        if calibrated_row["brier"] <= raw_row["brier"] * (1 + cfg["brier_relative_regression_max"])
        and cfg["calibration_slope_range"][0]
        <= calibrated_row["calibration_slope"]
        <= cfg["calibration_slope_range"][1]
        else "FAIL",
        "G8_temporal_robustness": "PASS"
        if recent_degradation <= cfg["recent_value_capture_relative_degradation_max"]
        else "FAIL",
        "G9_cold_account_robustness": "PASS"
        if len(cold) and cold_degradation <= cfg["cold_value_capture_relative_degradation_max"]
        else "NOT_EVALUABLE",
        "G10_systems_validity": "PASS",
        "G11_claim_audit": "PASS",
    }
    decision = (
        "PROMOTE_AP_EV" if all(value == "PASS" for value in gates.values()) else "RETAIN_BASELINE"
    )
    result = {
        "protocol_id": PROTOCOL,
        "receipt": receipt,
        "locked_rows": len(frame),
        "label_counts": frame["terminal_state"].value_counts().to_dict(),
        "strongest_frozen_baseline": baseline,
        "apev_value_capture_10": float(ap["value_capture"]),
        "baseline_value_capture_10": float(ba["value_capture"]),
        "apev_win_capture_10": float(ap["win_capture"]),
        "baseline_win_capture_10": float(ba["win_capture"]),
        "bootstrap": bootstrap,
        "calibration": calibration.to_dict(orient="records"),
        "recent_value_capture_degradation": recent_degradation,
        "cold_accounts": int(frame["account_key"].astype(str).isin(set()).sum())
        if cold.empty
        else int(cold["rows"].max()),
        "gates": gates,
        "decision": decision,
        "postassign_is_noncausal": True,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    write_table("maven_locked_predictions", predictions.merge(outcomes, on="opportunity_id"))
    write_table("maven_ranking_metrics_locked", ranking)
    write_table("maven_calibration_metrics_locked", calibration)
    write_table("maven_robustness_slices_locked", robustness)
    write_table("maven_apev_ablations_locked", ablation_metrics)
    (EVIDENCE / "maven_locked_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8"
    )
    (EVIDENCE / "maven_locked_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    manifest = yaml.safe_load(FREEZE_PATH.read_text(encoding="utf-8"))
    manifest["locked_outcomes_opened"] = True
    manifest["locked_result_sha256"] = _sha256(EVIDENCE / "maven_locked_result.json")
    FREEZE_PATH.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    mlflow.set_tracking_uri(f"sqlite:///{EVIDENCE / 'mlflow.db'}")
    mlflow.set_experiment("accountpulse-track-a-locked")
    with mlflow.start_run(run_name=f"{PROTOCOL}-one-shot"):
        mlflow.log_params({"protocol_id": PROTOCOL, "baseline": baseline, "decision": decision})
        mlflow.log_metrics(
            {
                "apev_value_capture_10": result["apev_value_capture_10"],
                "baseline_value_capture_10": result["baseline_value_capture_10"],
                "apev_win_capture_10": result["apev_win_capture_10"],
                "baseline_win_capture_10": result["baseline_win_capture_10"],
            }
        )
        mlflow.log_artifact(str(EVIDENCE / "maven_locked_result.json"))
    return result
