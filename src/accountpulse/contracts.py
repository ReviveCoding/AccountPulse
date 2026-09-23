"""Point-in-time, label, split, ranking, and capacity contracts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd


class Maturity(StrEnum):
    MATURE_POSITIVE = "MATURE_POSITIVE"
    MATURE_NEGATIVE = "MATURE_NEGATIVE"
    CENSORED = "CENSORED"


class TerminalState(StrEnum):
    OPEN = "OPEN"
    WON = "WON"
    LOST = "LOST"


def assert_point_in_time(frame: pd.DataFrame) -> None:
    """Ensure every feature was available no later than its prediction timestamp."""
    required = {"availability_timestamp", "prediction_timestamp"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing PIT columns: {sorted(missing)}")
    availability = pd.to_datetime(frame["availability_timestamp"], utc=True)
    prediction = pd.to_datetime(frame["prediction_timestamp"], utc=True)
    invalid = availability > prediction
    if invalid.any():
        raise ValueError(f"PIT violation in {int(invalid.sum())} row(s)")


def fixed_horizon_maturity(
    event_time_days: pd.Series,
    event_observed: pd.Series,
    followup_days: pd.Series,
    horizon_days: int,
) -> pd.Series:
    """Create explicit positive/negative/censored fixed-horizon states."""
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    event_time = pd.to_numeric(event_time_days)
    observed = event_observed.astype(bool)
    followup = pd.to_numeric(followup_days)
    positive = observed & (event_time <= horizon_days)
    mature = followup >= horizon_days
    values = np.where(
        positive,
        Maturity.MATURE_POSITIVE,
        np.where(mature, Maturity.MATURE_NEGATIVE, Maturity.CENSORED),
    )
    return pd.Series(values, index=event_time_days.index, dtype="string")


def validate_competing_states(states: pd.Series, event_observed: pd.Series) -> None:
    """Validate that Won and Lost are observed terminal events, not censoring."""
    normalized = states.astype("string").str.upper()
    allowed = {member.value for member in TerminalState}
    unknown = set(normalized.dropna().unique()) - allowed
    if unknown:
        raise ValueError(f"Unknown terminal states: {sorted(unknown)}")
    observed = event_observed.astype(bool)
    expected = normalized.isin([TerminalState.WON, TerminalState.LOST])
    if not bool((observed == expected).all()):
        raise ValueError("Observed flag must be true for both WON and LOST, false for OPEN")


def validate_qids(qids: Iterable[object]) -> None:
    """Require contiguous, non-singleton ranking groups."""
    series = pd.Series(list(qids), dtype="string")
    if series.empty or series.isna().any():
        raise ValueError("Ranking qids must be non-empty and non-null")
    seen: set[str] = set()
    previous: str | None = None
    counts = series.value_counts()
    if (counts < 2).any():
        raise ValueError("Every ranking qid must contain at least two rows")
    for qid in series:
        if qid != previous and qid in seen:
            raise ValueError("Ranking qids must form contiguous groups")
        seen.add(qid)
        previous = qid


@dataclass(frozen=True)
class CapacityResult:
    capacity: float
    selected_count: int
    value_capture: float
    win_capture: float
    realized_value: float
    oracle_regret: float


def capacity_metrics(
    score: np.ndarray,
    value: np.ndarray,
    won: np.ndarray,
    capacity: float,
) -> CapacityResult:
    """Compute deterministic capacity metrics with stable tie-breaking."""
    if not 0 < capacity <= 1:
        raise ValueError("capacity must be in (0, 1]")
    arrays = [np.asarray(x) for x in (score, value, won)]
    if len({len(x) for x in arrays}) != 1 or len(arrays[0]) == 0:
        raise ValueError("score, value, and won must be non-empty and equally sized")
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("capacity inputs must be finite")
    n_selected = max(1, int(np.ceil(len(arrays[0]) * capacity)))
    order = np.argsort(-arrays[0], kind="stable")[:n_selected]
    selected_value = float(arrays[1][order].sum())
    total_value = float(arrays[1].sum())
    total_wins = float(arrays[2].sum())
    oracle_value = float(np.sort(arrays[1])[-n_selected:].sum())
    return CapacityResult(
        capacity=capacity,
        selected_count=n_selected,
        value_capture=selected_value / total_value if total_value > 0 else 0.0,
        win_capture=float(arrays[2][order].sum()) / total_wins if total_wins > 0 else 0.0,
        realized_value=selected_value,
        oracle_regret=oracle_value - selected_value,
    )
