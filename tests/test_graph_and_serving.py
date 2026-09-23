from __future__ import annotations

import pandas as pd

from accountpulse.cli import score_uci_replay
from accountpulse.graph import graph_at_cutoff


def test_future_graph_edges_are_excluded() -> None:
    cutoff = pd.Timestamp("2020-02-01", tz="UTC")
    lines = pd.DataFrame(
        {
            "invoice_date": [
                pd.Timestamp("2020-01-01", tz="UTC"),
                pd.Timestamp("2020-03-01", tz="UTC"),
            ],
            "cancelled": [False, False],
            "quantity": [1, 1],
            "customer_id": ["c", "c"],
            "stock_code": ["past", "future"],
            "invoice": ["i1", "i2"],
            "line_value": [10.0, 100.0],
        }
    )
    snapshots = pd.DataFrame({"cutoff": [cutoff], "customer_id": ["c"], "purchase_next_90d": [1]})
    graph = graph_at_cutoff(lines, snapshots, cutoff)
    assert graph.features.shape[0] == 2  # one customer plus only the past product


def test_batch_replay_schema_and_capacity() -> None:
    result = score_uci_replay("2011-09-01", 0.10, None)
    expected = {
        "opportunity_id",
        "account_id",
        "win_probability",
        "expected_terminal_days",
        "expected_value",
        "accountpulse_ev_score",
        "rank",
        "p_top10",
        "review_flag",
        "model_version",
    }
    assert expected <= set(result.columns)
    assert result["selected"].sum() == 532
