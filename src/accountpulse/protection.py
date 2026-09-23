"""Fail-closed scientific freeze and LOCKED-access controls."""

from __future__ import annotations

import yaml

from accountpulse.paths import REPO


class ProtectedAccessError(RuntimeError):
    """Raised when scientific protected state forbids an operation."""


def assert_locked_access() -> None:
    manifest_path = REPO / "FREEZE_MANIFEST.yaml"
    receipt = REPO / "LOCKED_START_RECEIPT.json"
    if not manifest_path.exists():
        raise ProtectedAccessError("LOCKED access denied: freeze manifest is missing")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN" or not manifest.get("locked_evaluation_authorized"):
        raise ProtectedAccessError("LOCKED access denied: protocol is not FROZEN and authorized")
    if not receipt.exists():
        raise ProtectedAccessError("LOCKED access denied: start receipt is missing")


def refuse_force(force: bool) -> None:
    if force:
        raise ProtectedAccessError("--force cannot bypass scientific gates")
