"""UCI Online Retail II canonicalization and lifecycle baselines."""

from __future__ import annotations

import json
import time
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines import CoxPHFitter
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from accountpulse.contracts import capacity_metrics
from accountpulse.evidence import EVIDENCE, write_table
from accountpulse.gpu import GPUSlot
from accountpulse.paths import REPO

SEED = 20260922
NUMERIC = [
    "recency_days",
    "frequency",
    "monetary",
    "mean_basket",
    "mean_item_count",
    "product_breadth",
    "cancellation_rate",
    "history_days",
]
CATEGORICAL = ["country"]
FEATURES = NUMERIC + CATEGORICAL


def load_lines() -> pd.DataFrame:
    path = REPO / "data/incoming/uci/online_retail_II.xlsx"
    pieces = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    frame = pd.concat(pieces.values(), ignore_index=True)
    frame.columns = [
        "invoice",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "price",
        "customer_id",
        "country",
    ]
    frame["invoice_date"] = pd.to_datetime(frame["invoice_date"], utc=True)
    frame["customer_id"] = frame["customer_id"].astype("string")
    frame = frame.dropna(subset=["invoice", "invoice_date", "customer_id"])
    frame["invoice"] = frame["invoice"].astype("string")
    frame["stock_code"] = frame["stock_code"].astype("string")
    frame["description"] = frame["description"].astype("string")
    frame["line_value"] = frame["quantity"] * frame["price"]
    frame["cancelled"] = frame["invoice"].str.startswith("C") | (frame["quantity"] < 0)
    return frame


def build_baskets(lines: pd.DataFrame) -> pd.DataFrame:
    baskets = (
        lines.groupby(["invoice", "customer_id", "invoice_date", "country"], dropna=False)
        .agg(
            basket_spend=("line_value", "sum"),
            item_count=("quantity", lambda values: float(values.abs().sum())),
            distinct_products=("stock_code", "nunique"),
            cancellation_signal=("cancelled", "max"),
        )
        .reset_index()
        .sort_values(["invoice_date", "customer_id", "invoice"])
    )
    baskets["positive_spend"] = baskets["basket_spend"].clip(lower=0)
    return baskets


def lifecycle_snapshots(lines: pd.DataFrame, baskets: pd.DataFrame) -> pd.DataFrame:
    minimum = baskets["invoice_date"].min()
    month_start = pd.Timestamp(minimum.year, minimum.month, 1, tz="UTC")
    start = month_start + pd.DateOffset(months=5)
    end = baskets["invoice_date"].max().floor("D") - pd.Timedelta(days=90)
    cutoffs = pd.date_range(start, end, freq="MS")
    first_seen = baskets.groupby("customer_id")["invoice_date"].min()
    outputs: list[pd.DataFrame] = []
    for cutoff in cutoffs:
        history = baskets[baskets["invoice_date"] < cutoff]
        future = baskets[
            (baskets["invoice_date"] >= cutoff)
            & (baskets["invoice_date"] < cutoff + pd.Timedelta(days=90))
        ]
        if history.empty:
            continue
        group = history.groupby("customer_id")
        snapshot = group.agg(
            last_purchase=("invoice_date", "max"),
            first_purchase=("invoice_date", "min"),
            frequency=("invoice", "nunique"),
            monetary=("positive_spend", "sum"),
            mean_basket=("positive_spend", "mean"),
            mean_item_count=("item_count", "mean"),
            cancellation_rate=("cancellation_signal", "mean"),
            country=("country", "last"),
        ).reset_index()
        prior_lines = lines[lines["invoice_date"] < cutoff]
        breadth = prior_lines.groupby("customer_id")["stock_code"].nunique()
        snapshot["product_breadth"] = snapshot["customer_id"].map(breadth).fillna(0)
        snapshot["recency_days"] = (cutoff - snapshot["last_purchase"]).dt.total_seconds() / 86400
        snapshot["history_days"] = (cutoff - snapshot["first_purchase"]).dt.total_seconds() / 86400
        future_group = future.groupby("customer_id").agg(
            future_90d_spend=("positive_spend", "sum"),
            next_purchase=("invoice_date", "min"),
        )
        snapshot["future_90d_spend"] = (
            snapshot["customer_id"].map(future_group["future_90d_spend"]).fillna(0.0)
        )
        snapshot["purchase_next_90d"] = (snapshot["future_90d_spend"] > 0).astype(int)
        snapshot["time_to_next_days"] = (
            snapshot["customer_id"].map(future_group["next_purchase"]) - cutoff
        ).dt.total_seconds() / 86400
        snapshot["duration_days"] = snapshot["time_to_next_days"].fillna(90.0).clip(0.01, 90)
        snapshot["event_observed"] = snapshot["time_to_next_days"].notna().astype(int)
        snapshot["cutoff"] = cutoff
        snapshot["first_seen"] = snapshot["customer_id"].map(first_seen)
        outputs.append(snapshot)
    result = pd.concat(outputs, ignore_index=True)
    unique_cutoffs = sorted(result["cutoff"].unique())
    test_cutoff = pd.Timestamp(unique_cutoffs[-1])
    validation_cutoff = test_cutoff - pd.DateOffset(months=3)
    fit_max = validation_cutoff - pd.DateOffset(months=3)
    result["split"] = "PURGED"
    result.loc[result["cutoff"] <= fit_max, "split"] = "FIT"
    result.loc[result["cutoff"] == validation_cutoff, "split"] = "VALIDATION"
    result.loc[result["cutoff"] == test_cutoff, "split"] = "TEST"
    result["customer_cohort"] = np.where(result["first_seen"] <= fit_max, "KNOWN", "UNSEEN")
    return result


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)]
    )


def _metric_rows(
    model: str,
    frame: pd.DataFrame,
    score: np.ndarray,
    probability: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cohort in ("ALL", "KNOWN", "UNSEEN"):
        mask = (
            np.ones(len(frame), dtype=bool)
            if cohort == "ALL"
            else frame["customer_cohort"].eq(cohort).to_numpy()
        )
        if mask.sum() < 20 or frame.loc[mask, "purchase_next_90d"].nunique() < 2:
            continue
        y = frame.loc[mask, "purchase_next_90d"].to_numpy()
        value = frame.loc[mask, "future_90d_spend"].to_numpy(dtype=float)
        cohort_score = score[mask]
        cap = capacity_metrics(cohort_score, value, y, 0.10)
        rows.append(
            {
                "track": "C",
                "model": model,
                "split": "TEST",
                "cohort": cohort,
                "roc_auc": roc_auc_score(y, cohort_score),
                "brier": brier_score_loss(y, cohort_score) if probability else None,
                "log_loss": log_loss(y, np.clip(cohort_score, 1e-7, 1 - 1e-7))
                if probability
                else None,
                "value_capture_10": cap.value_capture,
                "win_capture_10": cap.win_capture,
                "selected_count": cap.selected_count,
            }
        )
    return rows


def run() -> dict[str, Any]:
    started = time.perf_counter()
    canonical_lines = REPO / "data/processed/uci_lines.parquet"
    canonical_baskets = REPO / "data/processed/uci_baskets.parquet"
    if canonical_lines.exists() and canonical_baskets.exists():
        lines = pd.read_parquet(canonical_lines)
        baskets = pd.read_parquet(canonical_baskets)
    else:
        lines = load_lines()
        baskets = build_baskets(lines)
        canonical_lines.parent.mkdir(parents=True, exist_ok=True)
        lines.to_parquet(canonical_lines, index=False)
        baskets.to_parquet(canonical_baskets, index=False)
    snapshots = lifecycle_snapshots(lines, baskets)
    fit = snapshots["split"] == "FIT"
    validation = snapshots["split"] == "VALIDATION"
    test = snapshots["split"] == "TEST"
    train = snapshots.loc[fit]
    test_frame = snapshots.loc[test].reset_index(drop=True)
    y_fit = train["purchase_next_90d"].to_numpy()
    logistic = Pipeline(
        [
            ("features", _preprocessor()),
            ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
        ]
    )
    logistic.fit(train[FEATURES], y_fit)
    logistic_score = logistic.predict_proba(test_frame[FEATURES])[:, 1]

    encoder = _preprocessor()
    fit_matrix = encoder.fit_transform(train[FEATURES])
    validation_matrix = encoder.transform(snapshots.loc[validation, FEATURES])
    test_matrix = encoder.transform(test_frame[FEATURES])
    with GPUSlot(timeout=10):
        boosted = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cuda",
            random_state=SEED,
            n_jobs=1,
        )
        boosted.fit(
            fit_matrix,
            y_fit,
            eval_set=[(validation_matrix, snapshots.loc[validation, "purchase_next_90d"])],
            verbose=False,
        )
    xgb_score = boosted.predict_proba(test_matrix)[:, 1]
    rfm_score = (
        -test_frame["recency_days"].rank(pct=True)
        + test_frame["frequency"].rank(pct=True)
        + test_frame["monetary"].rank(pct=True)
    ).to_numpy()

    cox_columns = [*NUMERIC, "duration_days", "event_observed"]
    cox_data = train[cox_columns].replace([np.inf, -np.inf], np.nan).dropna()
    cox = CoxPHFitter(penalizer=0.1)
    cox.fit(cox_data, duration_col="duration_days", event_col="event_observed")
    cox_risk = cox.predict_partial_hazard(test_frame[NUMERIC]).to_numpy()

    metrics: list[dict[str, Any]] = []
    metrics += _metric_rows("G0_rfm", test_frame, rfm_score, False)
    metrics += _metric_rows("G0_logistic", test_frame, logistic_score, True)
    metrics += _metric_rows("G1_xgboost_cuda", test_frame, xgb_score, True)
    metrics += _metric_rows("G2_cox_risk", test_frame, cox_risk, False)
    predictions = test_frame[
        ["customer_id", "cutoff", "customer_cohort", "purchase_next_90d", "future_90d_spend"]
    ].assign(
        rfm_score=rfm_score,
        logistic_score=logistic_score,
        xgboost_score=xgb_score,
        cox_risk=cox_risk,
    )
    write_table("uci_baskets", baskets)
    write_table("uci_lifecycle_snapshots", snapshots)
    write_table("uci_predictions", predictions)
    ranking_path = EVIDENCE / "ranking_metrics.parquet"
    prior = pd.read_parquet(ranking_path) if ranking_path.exists() else pd.DataFrame()
    if not prior.empty and "track" in prior:
        prior = prior.loc[prior["track"] != "C"]
    write_table("ranking_metrics", pd.concat([prior, pd.DataFrame(metrics)], ignore_index=True))
    model_dir = REPO / "artifacts/models/uci"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(logistic, model_dir / "rfm_logistic.joblib")
    joblib.dump((encoder, boosted), model_dir / "xgboost_cuda.joblib")
    joblib.dump(cox, model_dir / "cox.joblib")
    mlflow.set_tracking_uri(f"sqlite:///{REPO / 'artifacts/evidence/mlflow.db'}")
    mlflow.set_experiment("AccountPulse-E22-UCI-Baselines")
    with mlflow.start_run(run_name="uci_lifecycle_baselines"):
        mlflow.log_params({"seed": SEED, "horizon_days": 90, "features": FEATURES})
        for index, row in enumerate(metrics):
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"row{index}.{key}", float(value))
    result = {
        "status": "COMPLETE",
        "lines": len(lines),
        "baskets": len(baskets),
        "snapshots": len(snapshots),
        "test_rows": int(test.sum()),
        "runtime_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "claim_boundary": "PUBLIC_REAL_WHOLESALE_HEAVY_RETAIL_NOT_CRM",
    }
    (EVIDENCE / "uci_baseline_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
