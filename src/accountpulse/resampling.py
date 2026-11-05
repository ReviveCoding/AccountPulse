"""Row-position groups for normalized account and calendar identifiers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def grouped_row_positions(keys: np.ndarray) -> dict[str, np.ndarray]:
    """Keep first-seen string key order and original row order within each group."""
    return pd.Series(keys).groupby(keys, sort=False).indices
