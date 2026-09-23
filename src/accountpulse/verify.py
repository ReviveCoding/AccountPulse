"""Fast evidence and protection verification used by the pipeline."""

from __future__ import annotations

import json

import pandas as pd
import yaml

from accountpulse.evidence import EVIDENCE
from accountpulse.paths import REPO
from accountpulse.provenance import validate_criteo_source


def run() -> dict[str, object]:
    required = [
        "ranking_metrics.parquet",
        "capacity_curves.parquet",
        "calibration_metrics.parquet",
        "bootstrap_results.parquet",
        "robustness_slices.parquet",
        "systems_metrics.parquet",
        "causal_predictions.parquet",
        "causal_policy_metrics.parquet",
        "causal_policy_comparison.parquet",
        "causal_uplift_curves.parquet",
        "maven_ranking_metrics_locked.parquet",
        "maven_calibration_metrics_locked.parquet",
        "maven_robustness_slices_locked.parquet",
        "maven_ranking_uncertainty_locked.parquet",
        "maven_ranking_secondary_locked.parquet",
        "maven_survival_metrics_locked.parquet",
        "maven_value_metrics_locked.parquet",
        "maven_monitoring_metrics.parquet",
        "maven_shadow_replay.parquet",
    ]
    missing = [name for name in required if not (EVIDENCE / name).exists()]
    if missing:
        raise RuntimeError(f"Missing canonical evidence: {missing}")
    predictions = pd.read_parquet(EVIDENCE / "uci_predictions.parquet")
    numeric = predictions.select_dtypes(include="number")
    if numeric.isin([float("inf"), float("-inf")]).any().any():
        raise RuntimeError("Non-finite prediction evidence")
    receipt = json.loads((REPO / "LOCKED_START_RECEIPT.json").read_text(encoding="utf-8"))
    freeze = yaml.safe_load((REPO / "FREEZE_MANIFEST.yaml").read_text(encoding="utf-8"))
    if receipt["protocol_id"] != "AP-V1-TRACKA-20260923-A1":
        raise RuntimeError("Unexpected Track-A LOCKED receipt protocol")
    if freeze["status"] != "FROZEN" or freeze["locked_outcomes_opened"]:
        raise RuntimeError("Track-A freeze/LOCKED state is inconsistent")
    if (
        receipt["freeze_manifest_sha256"]
        != __import__("hashlib").sha256((REPO / "FREEZE_MANIFEST.yaml").read_bytes()).hexdigest()
    ):
        raise RuntimeError("Track-A immutable freeze hash differs from receipt")
    tracka = json.loads((EVIDENCE / "maven_locked_result.json").read_text(encoding="utf-8"))
    if tracka["decision"] != "RETAIN_BASELINE":
        raise RuntimeError("Track-A frozen decision changed")
    source_hash = validate_criteo_source(
        REPO / "data/incoming/criteo/criteo-research-uplift-v2.1.csv.gz"
    )
    causal = pd.read_parquet(EVIDENCE / "causal_predictions.parquet")
    if causal["row_id"].duplicated().any():
        raise RuntimeError("Track-D TEST row identities are not disjoint/unique")
    if set(causal["protocol_id"].astype(str)) != {"AP-V1-CAUSAL-BRIDGE-20260922-D2"}:
        raise RuntimeError("Unexpected causal protocol identity")
    result = {
        "status": "COMPLETE",
        "canonical_tables": required,
        "locked_outcomes_opened": True,
        "protocol_id": "AP-V1-PROTOCOL-20260922-R2",
        "track_a_protocol_id": "AP-V1-TRACKA-20260923-A1",
        "track_a_decision": "RETAIN_BASELINE",
        "causal_protocol_id": "AP-V1-CAUSAL-BRIDGE-20260922-D2",
        "criteo_source_sha256": source_hash,
    }
    (EVIDENCE / "verification_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
