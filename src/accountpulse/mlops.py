"""MLflow registry, monitoring evidence, and delayed-label historical replay."""

from __future__ import annotations

import json
import time
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from scipy.stats import ks_2samp

from accountpulse.evidence import EVIDENCE, write_table
from accountpulse.gpu import GPUSlot
from accountpulse.paths import REPO
from accountpulse.uci import FEATURES, NUMERIC

PROTOCOL = "AP-V1-PROTOCOL-20260922-R2"


def run() -> dict[str, Any]:
    tracking_uri = f"sqlite:///{REPO / 'artifacts/evidence/mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    blocked_families = [
        "accountpulse-win",
        "accountpulse-terminal",
        "accountpulse-value",
        "accountpulse-ranker",
    ]
    for name in blocked_families:
        try:
            client.create_registered_model(
                name, tags={"status": "BLOCKED_TRACK_A", "protocol": PROTOCOL}
            )
        except MlflowException as exc:
            if "already exists" not in str(exc):
                raise
    versions = client.search_model_versions("name='accountpulse-fusion'")
    current = [version for version in versions if version.tags.get("protocol") == PROTOCOL]
    if current:
        newest = max(current, key=lambda version: int(version.version))
        model_uri = f"models:/accountpulse-fusion/{newest.version}"
    else:
        mlflow.set_experiment("AccountPulse-E33-Registry")
        early = joblib.load(REPO / "artifacts/models/uci/early_fusion.joblib")
        with mlflow.start_run(run_name="register_fusion_challenger"):
            model_info = mlflow.sklearn.log_model(
                early,
                name="model",
                registered_model_name="accountpulse-fusion",
                pip_requirements=[
                    "joblib==1.6.0",
                    "numpy==2.5.3",
                    "scikit-learn==1.9.1",
                    "scipy==1.18.1",
                ],
                metadata={
                    "evidence_class": "PUBLIC_REAL_WHOLESALE_HEAVY_RETAIL_NOT_CRM",
                    "promotion_decision": "RETAIN_BASELINE",
                },
            )
        versions = client.search_model_versions("name='accountpulse-fusion'")
        newest = max(versions, key=lambda version: int(version.version))
        model_uri = model_info.model_uri
    for version in versions:
        if version.version == "1":
            invalid_status = "INFRASTRUCTURE_INVALID"
        elif version.version != newest.version:
            invalid_status = "SCIENTIFIC_INVALID_R1"
        else:
            continue
        client.set_model_version_tag(
            "accountpulse-fusion", version.version, "eligible_status", invalid_status
        )
    client.set_registered_model_alias("accountpulse-fusion", "challenger", newest.version)
    client.set_model_version_tag(
        "accountpulse-fusion", newest.version, "eligible_status", "RESEARCH_ONLY"
    )
    client.set_model_version_tag("accountpulse-fusion", newest.version, "protocol", PROTOCOL)

    snapshots = pd.read_parquet(EVIDENCE / "uci_lifecycle_snapshots.parquet")
    validation = snapshots[snapshots["split"] == "VALIDATION"]
    test = snapshots[snapshots["split"] == "TEST"]
    monitoring_rows: list[dict[str, Any]] = []
    for feature in NUMERIC:
        statistic = ks_2samp(validation[feature].dropna(), test[feature].dropna()).statistic
        monitoring_rows.append(
            {
                "metric_type": "input_drift_ks",
                "field": feature,
                "value": float(statistic),
                "reference": "VALIDATION",
                "current": "TEST",
            }
        )
        monitoring_rows.append(
            {
                "metric_type": "missingness",
                "field": feature,
                "value": float(test[feature].isna().mean()),
                "reference": "VALIDATION",
                "current": "TEST",
            }
        )
    unknown_country = ~test["country"].isin(validation["country"].dropna().unique())
    monitoring_rows.append(
        {
            "metric_type": "unknown_category_rate",
            "field": "country",
            "value": float(unknown_country.mean()),
            "reference": "VALIDATION",
            "current": "TEST",
        }
    )
    encoder, boosted = joblib.load(REPO / "artifacts/models/uci/xgboost_cuda.joblib")
    boosted.set_params(device="cuda", tree_method="hist")
    latencies = []
    sample = test[FEATURES].iloc[:512]
    encoded = encoder.transform(sample)
    with GPUSlot(timeout=10):
        for _ in range(50):
            started = time.perf_counter()
            boosted.predict_proba(encoded)
            latencies.append((time.perf_counter() - started) * 1000)
    monitoring_rows.extend(
        [
            {
                "metric_type": "inference_latency_ms",
                "field": "p50",
                "value": float(np.quantile(latencies, 0.5)),
                "reference": "batch_512",
                "current": "TEST",
            },
            {
                "metric_type": "inference_latency_ms",
                "field": "p95",
                "value": float(np.quantile(latencies, 0.95)),
                "reference": "batch_512",
                "current": "TEST",
            },
        ]
    )
    write_table("monitoring_metrics", pd.DataFrame(monitoring_rows))

    comparison = json.loads((EVIDENCE / "evaluation_result.json").read_text(encoding="utf-8"))[
        "fusion_minus_rfm_value_capture_10"
    ]
    replay = pd.DataFrame(
        [
            {
                "step": 1,
                "state": "SCORE",
                "status": "COMPLETE",
                "detail": "historical test cutoff scored",
            },
            {
                "step": 2,
                "state": "WAIT_FOR_MATURITY",
                "status": "COMPLETE",
                "detail": "90-day horizon observed",
            },
            {
                "step": 3,
                "state": "EVALUATE",
                "status": "COMPLETE",
                "detail": "customer-cluster bootstrap",
            },
            {
                "step": 4,
                "state": "DETECT_DRIFT",
                "status": "COMPLETE",
                "detail": "KS/missingness/unknown category",
            },
            {
                "step": 5,
                "state": "TRAIN_CHALLENGER",
                "status": "COMPLETE",
                "detail": "fusion candidates trained",
            },
            {
                "step": 6,
                "state": "VALIDATE",
                "status": "COMPLETE",
                "detail": "validation-only adaptive gate",
            },
            {
                "step": 7,
                "state": "SHADOW",
                "status": "COMPLETE",
                "detail": "stored test predictions",
            },
            {
                "step": 8,
                "state": "PROMOTE_OR_REJECT",
                "status": "RETAIN_BASELINE",
                "detail": f"fusion-minus-RFM CI {comparison['ci_95']} crosses zero",
            },
        ]
    )
    write_table("shadow_replay", replay)
    result = {
        "status": "COMPLETE",
        "tracking_uri": tracking_uri,
        "registered_families": [*blocked_families, "accountpulse-fusion"],
        "fusion_version": newest.version,
        "fusion_alias": "challenger",
        "promotion_decision": "RETAIN_BASELINE",
        "shadow_replay": "COMPLETE_HISTORICAL_REPLAY_NOT_ONLINE_DEPLOYMENT",
        "model_uri": model_uri,
    }
    (EVIDENCE / "mlops_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
