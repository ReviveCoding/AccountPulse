"""Validate final handoff documents and refresh canonical structured evidence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from accountpulse.evidence import write_table
from accountpulse.paths import REPO, STATE_PATH

PREDICTIVE_PROTOCOL = "AP-V1-PROTOCOL-20260922-R2"
CAUSAL_PROTOCOL = "AP-V1-CAUSAL-BRIDGE-20260922-D2"
TRACK_A_PROTOCOL = "AP-V1-TRACKA-20260923-A1"
MANDATORY_REPORTS = (
    "reports/AccountPulse_Technical_Report.md",
    "reports/AccountPulse_Executive_Summary.md",
    "reports/ACCOUNT_PULSE_FINAL_STATUS.md",
    "reports/adversarial_audit.md",
    "DATA_CARD.md",
    "MODEL_CARD.md",
    "DECISION_MEMO.md",
    "LIMITATIONS.md",
    "RESUME_EVIDENCE.md",
    "INTERVIEW_NOTES.md",
)


def _validate_report(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    prohibited_stale = (
        "Criteo causal work is externally blocked",
        "no randomized causal bridge was run",
        "Track D is blocked",
    )
    if any(phrase in text for phrase in prohibited_stale):
        raise ValueError(f"Stale Track-D blocker wording in {path}")


def run() -> dict[str, Any]:
    """Fail closed on stale claims and materialize YAML ledgers into DuckDB/Parquet."""
    for relative in MANDATORY_REPORTS:
        path = REPO / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        _validate_report(path)

    registry = yaml.safe_load((REPO / "EXPERIMENT_REGISTRY.yaml").read_text(encoding="utf-8"))
    if registry["experiments"]["E28"]["status"] != "COMPLETE_D2":
        raise ValueError("E28 must be COMPLETE_D2 before final reporting")
    experiments = [
        {
            "experiment_id": experiment_id,
            "status": definition["status"],
            "question": definition["question"],
            "claim_boundary": definition["claim_boundary"],
        }
        for experiment_id, definition in registry["experiments"].items()
    ]
    write_table("experiment_status", pd.DataFrame(experiments))

    claims = yaml.safe_load((REPO / "CLAIM_LEDGER.yaml").read_text(encoding="utf-8"))
    write_table("claim_ledger", pd.DataFrame(claims["claims"]))

    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if not state["protected_state"]["locked_outcomes_opened"]:
        raise ValueError("Track-A LOCKED state must be complete for the A1 final report")
    state["stages"]["report"] = {
        "status": "COMPLETE",
        "protocol_id": PREDICTIVE_PROTOCOL,
        "causal_protocol_id": CAUSAL_PROTOCOL,
        "track_a_protocol_id": TRACK_A_PROTOCOL,
    }
    state["updated_at"] = datetime.now(UTC).isoformat()
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "COMPLETE",
        "predictive_protocol": PREDICTIVE_PROTOCOL,
        "causal_protocol": CAUSAL_PROTOCOL,
        "track_a_protocol": TRACK_A_PROTOCOL,
        "decision": "RETAIN_BASELINE",
        "reports": list(MANDATORY_REPORTS),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
