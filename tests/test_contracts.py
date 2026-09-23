from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from accountpulse.contracts import (
    Maturity,
    assert_point_in_time,
    capacity_metrics,
    fixed_horizon_maturity,
    validate_competing_states,
    validate_qids,
)
from accountpulse.scoring import accountpulse_ev


def test_point_in_time_accepts_equal_and_prior() -> None:
    frame = pd.DataFrame(
        {
            "availability_timestamp": ["2020-01-01", "2020-01-02"],
            "prediction_timestamp": ["2020-01-01", "2020-01-03"],
        }
    )
    assert_point_in_time(frame)


def test_point_in_time_rejects_future() -> None:
    frame = pd.DataFrame(
        {"availability_timestamp": ["2020-01-02"], "prediction_timestamp": ["2020-01-01"]}
    )
    with pytest.raises(ValueError, match="PIT violation"):
        assert_point_in_time(frame)


def test_maturity_never_turns_immature_row_negative() -> None:
    result = fixed_horizon_maturity(
        pd.Series([20, 120, 120]),
        pd.Series([True, False, False]),
        pd.Series([20, 90, 40]),
        90,
    )
    assert result.tolist() == [
        Maturity.MATURE_POSITIVE,
        Maturity.MATURE_NEGATIVE,
        Maturity.CENSORED,
    ]


def test_lost_is_observed_competing_event() -> None:
    validate_competing_states(pd.Series(["WON", "LOST", "OPEN"]), pd.Series([1, 1, 0]))
    with pytest.raises(ValueError):
        validate_competing_states(pd.Series(["WON", "LOST"]), pd.Series([1, 0]))


def test_qids_must_be_contiguous_and_non_singleton() -> None:
    validate_qids(["a", "a", "b", "b"])
    with pytest.raises(ValueError, match="contiguous"):
        validate_qids(["a", "a", "b", "b", "a"])
    with pytest.raises(ValueError, match="at least two"):
        validate_qids(["a", "a", "b"])


def test_capacity_metrics_and_oracle_regret() -> None:
    result = capacity_metrics(
        np.array([0.9, 0.8, 0.1]),
        np.array([1.0, 5.0, 10.0]),
        np.array([1, 0, 1]),
        1 / 3,
    )
    assert result.selected_count == 1
    assert result.value_capture == pytest.approx(1 / 16)
    assert result.win_capture == pytest.approx(0.5)
    assert result.oracle_regret == 9


def test_accountpulse_ev_components() -> None:
    score = accountpulse_ev(
        np.array([[0.4, 0.6]]),
        np.array([[0.5, 0.5]]),
        np.array([[100.0, 200.0]]),
        np.array([30.0, 90.0]),
        horizon_days=60,
        rho=0,
    )
    assert score.tolist() == pytest.approx([20.0])
