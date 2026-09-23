"""Central paths with explicit WSL/cache contracts."""

from __future__ import annotations

import os
from pathlib import Path


def _repository_root() -> Path:
    """Resolve data-bearing repository independently of package install location."""
    configured = os.environ.get("ACCOUNT_PULSE_REPO")
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "artifacts/state/pipeline_state.json"
        ).is_file():
            return candidate.resolve()
    return Path(__file__).resolve().parents[2]


REPO = _repository_root()
RUNTIME = Path(os.environ.get("ACCOUNT_PULSE_RUNTIME", "/home/bjw-0/.local/share/accountpulse"))
SCRATCH = Path(os.environ.get("ACCOUNT_PULSE_SCRATCH", "/home/bjw-0/.cache/accountpulse"))
STATE_PATH = REPO / "artifacts/state/pipeline_state.json"
GPU_LOCK_PATH = SCRATCH / "locks/gpu.lock"


def ensure_runtime_dirs() -> None:
    """Create only the bounded AccountPulse runtime locations."""
    for path in (SCRATCH, SCRATCH / "locks", SCRATCH / "tmp"):
        path.mkdir(parents=True, exist_ok=True)
