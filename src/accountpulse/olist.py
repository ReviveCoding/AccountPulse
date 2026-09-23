"""PIT-safe Olist sparse funnel external-validation pipeline."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from accountpulse.contracts import capacity_metrics
from accountpulse.evidence import EVIDENCE, write_table
from accountpulse.gpu import GPUSlot
from accountpulse.paths import REPO

SEED = 20260922
FEATURES = ["origin", "landing_page_id", "first_contact_month"]


def _break_ties(score: np.ndarray, identifiers: pd.Series) -> np.ndarray:
    """Use an outcome-independent stable ID hash to order exact score ties."""
    hashes = np.array(
        [
            int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:8], "big")
            for value in identifiers
        ],
        dtype=np.float64,
    )
    hashes /= np.iinfo(np.uint64).max
    scale = max(1.0, float(np.max(np.abs(score)))) * 1e-12
    return np.asarray(score, dtype=float) + hashes * scale


def build_leads() -> pd.DataFrame:
    root = REPO / "data/incoming/olist"
    leads = pd.read_csv(root / "olist_marketing_qualified_leads_dataset.csv")
    deals = pd.read_csv(root / "olist_closed_deals_dataset.csv")
    orders = pd.read_csv(root / "olist_orders_dataset.csv")
    items = pd.read_csv(root / "olist_order_items_dataset.csv")
    leads["first_contact_date"] = pd.to_datetime(leads["first_contact_date"], utc=True)
    deals["won_date"] = pd.to_datetime(deals["won_date"], utc=True)
    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], utc=True
    )
    order_value = items.groupby(["order_id", "seller_id"], as_index=False)["price"].sum()
    commerce = order_value.merge(
        orders[["order_id", "order_purchase_timestamp"]], on="order_id", how="left", validate="m:1"
    )
    linkage = deals[["mql_id", "seller_id", "won_date"]].drop_duplicates("mql_id")
    frame = leads.merge(linkage, on="mql_id", how="left", validate="1:1")
    frame["converted"] = frame["seller_id"].notna().astype(int)
    frame["days_to_win"] = (
        frame["won_date"] - frame["first_contact_date"]
    ).dt.total_seconds() / 86400
    latest = orders["order_purchase_timestamp"].max()
    for horizon in (90, 180):
        joined = frame[["mql_id", "seller_id", "first_contact_date"]].merge(
            commerce, on="seller_id", how="left"
        )
        delta = (
            joined["order_purchase_timestamp"] - joined["first_contact_date"]
        ).dt.total_seconds() / 86400
        joined["eligible_value"] = joined["price"].where(delta.between(0, horizon), 0.0)
        values = joined.groupby("mql_id")["eligible_value"].sum()
        frame[f"value_{horizon}d"] = frame["mql_id"].map(values).fillna(0.0)
        frame[f"mature_{horizon}d"] = (
            frame["first_contact_date"] + pd.Timedelta(days=horizon) <= latest
        )
    frame["first_contact_month"] = frame["first_contact_date"].dt.month.astype("string")
    # No field from closed_deals is returned as a predictor.
    keep = [
        "mql_id",
        "first_contact_date",
        *FEATURES,
        "converted",
        "days_to_win",
        "value_90d",
        "mature_90d",
        "value_180d",
        "mature_180d",
    ]
    result = frame[keep].sort_values(["first_contact_date", "mql_id"]).reset_index(drop=True)
    return result


def split(frame: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(frame["first_contact_date"], utc=True)
    test_start = dates.max().normalize() - pd.Timedelta(days=29)
    validation_end = test_start - pd.Timedelta(days=90)
    validation_start = validation_end - pd.Timedelta(days=30)
    fit_end = validation_start - pd.Timedelta(days=90)
    labels = pd.Series("PURGED", index=frame.index, dtype="string")
    labels.loc[dates < fit_end] = "FIT"
    labels.loc[(dates >= validation_start) & (dates < validation_end)] = "VALIDATION"
    labels.loc[dates >= test_start] = "TEST"
    return labels


def _preprocessor() -> ColumnTransformer:
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer([("categorical", categorical, FEATURES)])


def _metrics(
    name: str,
    split_name: str,
    y: np.ndarray,
    score: np.ndarray,
    value: np.ndarray,
    *,
    probability: bool = True,
) -> dict[str, Any]:
    cap = capacity_metrics(score, value, y, 0.10)
    return {
        "track": "B",
        "model": name,
        "split": split_name,
        "roc_auc": roc_auc_score(y, score),
        "brier": brier_score_loss(y, score) if probability else None,
        "log_loss": log_loss(y, np.clip(score, 1e-7, 1 - 1e-7)) if probability else None,
        "value_capture_10": cap.value_capture,
        "win_capture_10": cap.win_capture,
        "selected_count": cap.selected_count,
    }


def run() -> dict[str, Any]:
    started = time.perf_counter()
    frame = build_leads()
    frame["split"] = split(frame)
    if not bool(frame["mature_90d"].all()):
        frame = frame.loc[frame["mature_90d"]].copy()
        frame["split"] = split(frame)
    fit = frame["split"] == "FIT"
    validation = frame["split"] == "VALIDATION"
    test = frame["split"] == "TEST"
    x_fit, y_fit = frame.loc[fit, FEATURES], frame.loc[fit, "converted"].to_numpy()
    x_val, y_val = frame.loc[validation, FEATURES], frame.loc[validation, "converted"].to_numpy()
    x_test, y_test = frame.loc[test, FEATURES], frame.loc[test, "converted"].to_numpy()
    value_test = frame.loc[test, "value_90d"].to_numpy(dtype=float)

    fit_frame = frame.loc[fit]
    global_value_rate = float(fit_frame["value_90d"].mean())
    origin_stats = fit_frame.groupby("origin")["value_90d"].agg(["sum", "count"])
    origin_value_prior = (origin_stats["sum"] + 20 * global_value_rate) / (
        origin_stats["count"] + 20
    )
    prior = (
        frame.loc[test, "origin"].map(origin_value_prior).fillna(global_value_rate).to_numpy(float)
    )
    logistic = Pipeline(
        [
            ("features", _preprocessor()),
            ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
        ]
    )
    logistic.fit(x_fit, y_fit)
    logistic_test = logistic.predict_proba(x_test)[:, 1]

    encoded_fit = _preprocessor()
    fit_matrix = encoded_fit.fit_transform(x_fit)
    val_matrix = encoded_fit.transform(x_val)
    test_matrix = encoded_fit.transform(x_test)
    with GPUSlot(timeout=10):
        boosted = xgb.XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cuda",
            random_state=SEED,
            n_jobs=1,
        )
        boosted.fit(fit_matrix, y_fit, eval_set=[(val_matrix, y_val)], verbose=False)
    boosted_test = boosted.predict_proba(test_matrix)[:, 1]

    # Compare raw, Platt, and isotonic using validation only.
    raw_val = boosted.predict_proba(val_matrix)[:, 1]
    raw_test = boosted_test
    val_logit = np.log(np.clip(raw_val, 1e-6, 1 - 1e-6) / np.clip(1 - raw_val, 1e-6, 1))
    test_logit = np.log(np.clip(raw_test, 1e-6, 1 - 1e-6) / np.clip(1 - raw_test, 1e-6, 1))
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(val_logit.reshape(-1, 1), y_val)
    platt_val = platt.predict_proba(val_logit.reshape(-1, 1))[:, 1]
    platt_test = platt.predict_proba(test_logit.reshape(-1, 1))[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(raw_val, y_val)
    isotonic_val = isotonic.predict(raw_val)
    isotonic_test = isotonic.predict(raw_test)
    calibration_candidates = {
        "raw": (raw_val, raw_test),
        "platt": (platt_val, platt_test),
        "isotonic": (isotonic_val, isotonic_test),
    }
    calibration_method = min(
        calibration_candidates,
        key=lambda name: brier_score_loss(y_val, calibration_candidates[name][0]),
    )
    calibrated_test = calibration_candidates[calibration_method][1]

    # Winner value and time use only training outcomes; no winner-only fields are predictors.
    winner_fit = fit & (frame["converted"] == 1)
    winner_matrix = encoded_fit.transform(frame.loc[winner_fit, FEATURES])
    with GPUSlot(timeout=10):
        value_model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.04,
            objective="reg:squarederror",
            tree_method="hist",
            device="cuda",
            random_state=SEED,
            n_jobs=1,
        ).fit(winner_matrix, np.log1p(frame.loc[winner_fit, "value_90d"].to_numpy(float)))
        time_model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.04,
            objective="reg:squarederror",
            tree_method="hist",
            device="cuda",
            random_state=SEED,
            n_jobs=1,
        ).fit(winner_matrix, frame.loc[winner_fit, "days_to_win"].fillna(0).to_numpy(float))
    val_conditional_value = np.expm1(value_model.predict(val_matrix)).clip(min=0)
    test_conditional_value = np.expm1(value_model.predict(test_matrix)).clip(min=0)
    val_days = time_model.predict(val_matrix).clip(min=0)
    test_days = time_model.predict(test_matrix).clip(min=0)
    rho_candidates = [0.0, 0.001, 0.005, 0.01]
    val_probability = calibration_candidates[calibration_method][0]
    val_value = frame.loc[validation, "value_90d"].to_numpy(dtype=float)
    rho = max(
        rho_candidates,
        key=lambda r: (
            capacity_metrics(
                val_probability * val_conditional_value * np.exp(-r * val_days),
                val_value,
                y_val,
                0.10,
            ).value_capture
        ),
    )
    simple_ev = calibrated_test * test_conditional_value
    sparse_apev = simple_ev * np.exp(-rho * test_days)
    test_ids = frame.loc[test, "mql_id"]
    prior = _break_ties(prior, test_ids)
    logistic_test = _break_ties(logistic_test, test_ids)
    boosted_test = _break_ties(boosted_test, test_ids)
    calibrated_test = _break_ties(calibrated_test, test_ids)
    simple_ev = _break_ties(simple_ev, test_ids)
    sparse_apev = _break_ties(sparse_apev, test_ids)
    metrics = [
        _metrics("business_prior", "TEST", y_test, prior, value_test, probability=False),
        _metrics("logistic", "TEST", y_test, logistic_test, value_test),
        _metrics("xgboost_cuda", "TEST", y_test, boosted_test, value_test),
        _metrics(f"xgboost_{calibration_method}", "TEST", y_test, calibrated_test, value_test),
        _metrics("simple_ev", "TEST", y_test, simple_ev, value_test, probability=False),
        _metrics(
            "sparse_accountpulse_ev", "TEST", y_test, sparse_apev, value_test, probability=False
        ),
    ]
    predictions = frame.loc[test, ["mql_id", "first_contact_date", "converted", "value_90d"]].copy()
    predictions = predictions.assign(
        split="TEST",
        prior_score=prior,
        logistic_score=logistic_test,
        xgboost_score=boosted_test,
        calibrated_score=calibrated_test,
        simple_ev_score=simple_ev,
        accountpulse_ev_score=sparse_apev,
    )
    write_table("olist_leads", frame)
    write_table("olist_predictions", predictions)
    ranking_path = EVIDENCE / "ranking_metrics.parquet"
    prior_metrics = pd.read_parquet(ranking_path) if ranking_path.exists() else pd.DataFrame()
    if not prior_metrics.empty and "track" in prior_metrics:
        prior_metrics = prior_metrics.loc[prior_metrics["track"] != "B"]
    write_table(
        "ranking_metrics", pd.concat([prior_metrics, pd.DataFrame(metrics)], ignore_index=True)
    )
    models = REPO / "artifacts/models/olist"
    models.mkdir(parents=True, exist_ok=True)
    joblib.dump(logistic, models / "logistic.joblib")
    joblib.dump((calibration_method, platt, isotonic), models / "xgboost_calibration.joblib")
    joblib.dump((encoded_fit, boosted), models / "xgboost_cuda.joblib")
    joblib.dump((value_model, time_model), models / "value_time_cuda.joblib")

    mlflow.set_tracking_uri(f"sqlite:///{REPO / 'artifacts/evidence/mlflow.db'}")
    mlflow.set_experiment("AccountPulse-E21-Olist")
    with mlflow.start_run(run_name="olist_sparse_external"):
        mlflow.log_params(
            {
                "seed": SEED,
                "horizon_days": 90,
                "rho": rho,
                "features": FEATURES,
                "calibration": calibration_method,
                "protocol_revision": "E21-R2",
            }
        )
        for row in metrics:
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{row['model']}.{key}", float(value))
        mlflow.log_artifact(str(REPO / "FEATURE_AVAILABILITY.csv"))
    result = {
        "status": "COMPLETE",
        "rows": len(frame),
        "test_rows": int(test.sum()),
        "rho": rho,
        "calibration_method": calibration_method,
        "protocol_revision": "AP-V1-PROTOCOL-20260922-R2",
        "runtime_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "claim_boundary": "PUBLIC_REAL_SPARSE_FUNNEL_EXTERNAL_METHOD_REPLICATION",
    }
    (REPO / "artifacts/evidence/olist_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
