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
from sklearn.pipeline import Pipeline

from accountpulse.evidence import EVIDENCE, write_table
from accountpulse.gpu import GPUSlot
from accountpulse.paths import REPO
from accountpulse.uci import FEATURES, NUMERIC

PROTOCOL = "AP-V1-PROTOCOL-20260922-R2"
TRACK_A_PROTOCOL = "AP-V1-TRACKA-20260923-A1"


def register_tracka() -> dict[str, Any]:
    """Register frozen eligible Track-A components as shadow models after retention."""
    from accountpulse.tracka_models import BUNDLE_PATH

    tracking_uri = f"sqlite:///{REPO / 'artifacts/evidence/mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("AccountPulse-E33-TrackA-Registry")
    client = MlflowClient(tracking_uri=tracking_uri)
    bundle = joblib.load(BUNDLE_PATH)
    families = {
        "accountpulse-win": Pipeline(
            [("features", bundle.preprocessor), ("model", bundle.classifiers["B4_xgboost_cuda"])]
        ),
        "accountpulse-terminal": Pipeline(
            [("features", bundle.preprocessor), ("model", bundle.timing_model)]
        ),
        "accountpulse-value": Pipeline(
            [("features", bundle.preprocessor), ("model", bundle.direct_value_model)]
        ),
        "accountpulse-ranker": Pipeline(
            [("features", bundle.preprocessor), ("model", bundle.ranker)]
        ),
    }
    versions: dict[str, str] = {}
    for name, model in families.items():
        existing = [
            version
            for version in client.search_model_versions(f"name='{name}'")
            if version.tags.get("protocol") == TRACK_A_PROTOCOL
        ]
        if existing:
            newest = max(existing, key=lambda version: int(version.version))
        else:
            with mlflow.start_run(run_name=f"register-{name}-a1"):
                mlflow.sklearn.log_model(
                    model,
                    name="model",
                    registered_model_name=name,
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                    metadata={
                        "protocol": TRACK_A_PROTOCOL,
                        "evidence_class": "FICTITIOUS_PUBLIC_CRM_NOT_PRODUCTION",
                        "promotion_decision": "RETAIN_BASELINE",
                    },
                )
            candidates = client.search_model_versions(f"name='{name}'")
            newest = max(candidates, key=lambda version: int(version.version))
        client.set_model_version_tag(name, newest.version, "protocol", TRACK_A_PROTOCOL)
        client.set_model_version_tag(name, newest.version, "eligible_status", "RESEARCH_SHADOW")
        client.set_registered_model_alias(name, "shadow", newest.version)
        client.set_registered_model_tag(name, "status", "COMPLETE_A1_RETAIN_BASELINE")
        versions[name] = newest.version
    result = {
        "protocol_id": TRACK_A_PROTOCOL,
        "status": "COMPLETE_LOCAL_REGISTRY",
        "versions": versions,
        "alias": "shadow",
        "decision": "RETAIN_BASELINE",
        "champion": "B1_business_heuristic_nonserialized",
    }
    (EVIDENCE / "tracka_registry_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


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
