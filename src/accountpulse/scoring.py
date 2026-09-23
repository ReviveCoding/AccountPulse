"""Interpretable AccountPulse-EV and direct score composition."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray


def accountpulse_ev(
    terminal_mass: np.ndarray,
    won_given_terminal: np.ndarray,
    value_given_won: np.ndarray,
    time_days: np.ndarray,
    horizon_days: int,
    rho: float,
) -> NDArray[np.float64]:
    """Compute the discrete AP-EV sum across eligible terminal times."""
    terminal_mass = np.asarray(terminal_mass, dtype=float)
    won_given_terminal = np.asarray(won_given_terminal, dtype=float)
    value_given_won = np.asarray(value_given_won, dtype=float)
    time_days = np.asarray(time_days, dtype=float)
    if (
        terminal_mass.shape != won_given_terminal.shape
        or terminal_mass.shape != value_given_won.shape
    ):
        raise ValueError("AP-EV component matrices must share shape")
    if terminal_mass.ndim != 2 or time_days.ndim != 1:
        raise ValueError("Components must be 2D and time_days must be 1D")
    if terminal_mass.shape[1] != len(time_days):
        raise ValueError("Time grid must match component columns")
    if horizon_days <= 0 or rho < 0:
        raise ValueError("horizon_days must be positive and rho non-negative")
    if not all(np.isfinite(x).all() for x in (terminal_mass, won_given_terminal, value_given_won)):
        raise ValueError("AP-EV components must be finite")
    if (
        (terminal_mass < 0).any()
        or (won_given_terminal < 0).any()
        or (won_given_terminal > 1).any()
    ):
        raise ValueError("Invalid probability component")
    mask = time_days <= horizon_days
    discount = np.exp(-rho * time_days[mask])
    score = (
        terminal_mass[:, mask] * won_given_terminal[:, mask] * value_given_won[:, mask] * discount
    ).sum(axis=1)
    return cast(NDArray[np.float64], np.asarray(score, dtype=float))
