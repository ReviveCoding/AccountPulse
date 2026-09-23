"""Fast evidence and protection verification used by the pipeline."""

from __future__ import annotations

import json

import pandas as pd

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
    ]
    missing = [name for name in required if not (EVIDENCE / name).exists()]
    if missing:
        raise RuntimeError(f"Missing canonical evidence: {missing}")
    predictions = pd.read_parquet(EVIDENCE / "uci_predictions.parquet")
    numeric = predictions.select_dtypes(include="number")
    if numeric.isin([float("inf"), float("-inf")]).any().any():
        raise RuntimeError("Non-finite prediction evidence")
    if (REPO / "LOCKED_START_RECEIPT.json").exists():
        raise RuntimeError("Unexpected LOCKED receipt while freeze is blocked")
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
        "locked_outcomes_opened": False,
        "protocol_id": "AP-V1-PROTOCOL-20260922-R2",
        "causal_protocol_id": "AP-V1-CAUSAL-BRIDGE-20260922-D2",
        "criteo_source_sha256": source_hash,
    }
    (EVIDENCE / "verification_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
