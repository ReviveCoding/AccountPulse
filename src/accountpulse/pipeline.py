"""Resume-safe pipeline runner with protected-stage enforcement."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml

from accountpulse.paths import REPO, STATE_PATH
from accountpulse.protection import ProtectedAccessError, assert_locked_access, refuse_force

STAGES: dict[str, tuple[str, str]] = {
    "qualify": ("accountpulse.gpu", "qualify"),
    "acquire": ("accountpulse.provenance", "write_manifest"),
    "eda": ("accountpulse.eda", "run"),
    "olist": ("accountpulse.olist", "run"),
    "olist_evaluate": ("accountpulse.olist_evaluate", "run"),
    "uci": ("accountpulse.uci", "run"),
    "temporal": ("accountpulse.temporal", "run"),
    "graph": ("accountpulse.graph", "run"),
    "text": ("accountpulse.text_model", "run"),
    "fusion": ("accountpulse.fusion", "run"),
    "evaluate": ("accountpulse.evaluate", "run"),
    "causal": ("accountpulse.causal", "run"),
    "mlops": ("accountpulse.mlops", "run"),
    "report": ("accountpulse.reporting", "run"),
    "verify": ("accountpulse.verify", "run"),
}


def load_state() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(STATE_PATH.read_text(encoding="utf-8")))


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(UTC).isoformat()
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE_PATH)


def resolve(stage: str) -> Callable[[], Any]:
    module_name, function_name = STAGES[stage]
    function = getattr(importlib.import_module(module_name), function_name)
    return function  # type: ignore[no-any-return]


def run_pipeline(
    config_path: Path,
    requested: list[str] | None,
    *,
    resume: bool,
    dry_run: bool,
    force: bool,
) -> dict[str, Any]:
    refuse_force(force)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selected = requested or config["stages"]
    unknown = set(selected) - set(STAGES) - {"freeze", "locked"}
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")
    state = load_state()
    plan = []
    for stage in selected:
        prior = state.setdefault("stages", {}).get(stage, {})
        action = "SKIP_COMPLETE" if resume and prior.get("status") == "COMPLETE" else "RUN"
        plan.append({"stage": stage, "action": action})
    if dry_run:
        return {"status": "DRY_RUN", "plan": plan}
    for item in plan:
        stage = item["stage"]
        if item["action"] == "SKIP_COMPLETE":
            continue
        if stage == "locked":
            assert_locked_access()
            raise ProtectedAccessError("LOCKED runner is unavailable until Track A is recovered")
        if stage == "freeze":
            raise ProtectedAccessError("Freeze denied while Track A is BLOCKED_EXTERNAL_DATA")
        state["current_stage"] = stage
        state["stages"][stage] = {
            "status": "RUNNING",
            "started_at": datetime.now(UTC).isoformat(),
        }
        save_state(state)
        try:
            result = resolve(stage)()
        except Exception as exc:
            state["stages"][stage] = {
                "status": "FAILED",
                "failure_class": "INFRASTRUCTURE_FAILURE",
                "error": repr(exc),
            }
            save_state(state)
            raise
        state["stages"][stage] = {
            "status": "COMPLETE",
            "completed_at": datetime.now(UTC).isoformat(),
            "result_type": type(result).__name__,
        }
        save_state(state)
    return {"status": "COMPLETE", "plan": plan}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m accountpulse.pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", type=Path, default=REPO / "configs/full.yaml")
    run_parser.add_argument("--stage", action="append", dest="stages")
    run_parser.add_argument("--no-resume", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run_pipeline(
        args.config,
        args.stages,
        resume=not args.no_resume,
        dry_run=args.dry_run,
        force=args.force,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
