"""Canonical Parquet evidence and DuckDB catalog helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from accountpulse.paths import REPO

EVIDENCE = REPO / "artifacts/evidence"
WAREHOUSE = EVIDENCE / "accountpulse.duckdb"


def write_table(name: str, frame: pd.DataFrame) -> Path:
    """Atomically replace a canonical evidence table and register a DuckDB view."""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE / f"{name}.parquet"
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    with duckdb.connect(str(WAREHOUSE)) as connection:
        escaped = str(path).replace("'", "''")
        connection.execute(
            f"CREATE OR REPLACE VIEW \"{name}\" AS SELECT * FROM read_parquet('{escaped}')"
        )
    return path


def append_table(name: str, rows: list[dict[str, Any]]) -> Path:
    """Append rows to a compact evidence table."""
    path = EVIDENCE / f"{name}.parquet"
    additions = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_parquet(path)
        additions = pd.concat([existing, additions], ignore_index=True)
    return write_table(name, additions)


def upsert_table(name: str, frame: pd.DataFrame, keys: list[str]) -> Path:
    """Replace matching evidence rows, making reruns idempotent on explicit keys."""
    path = EVIDENCE / f"{name}.parquet"
    if path.exists():
        existing = pd.read_parquet(path)
        if not existing.empty and all(key in existing for key in keys):
            incoming_keys = frame[keys].drop_duplicates()
            marked = existing.merge(incoming_keys.assign(_replace=True), on=keys, how="left")
            existing = marked.loc[marked["_replace"].isna(), existing.columns]
            frame = pd.concat([existing, frame], ignore_index=True)
    return write_table(name, frame)
