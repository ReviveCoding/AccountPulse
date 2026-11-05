import numpy as np
import pandas as pd

from accountpulse import olist_evaluate
from accountpulse.contracts import capacity_metrics


def test_cached_olist_scores_and_week_groups_preserve_temporal_interval(tmp_path, monkeypatch):
    rng = np.random.default_rng(51)
    n = 60
    frame = pd.DataFrame(
        {
            "converted": np.arange(n) % 2,
            "value_90d": rng.random(n) + 1,
            "first_contact_date": pd.date_range("2026-01-01", periods=n, freq="D"),
        }
    )
    for column in olist_evaluate.SCORES.values():
        frame[column] = rng.random(n)

    class Calibrator:
        def predict_proba(self, values):
            probability = np.full(len(values), 0.5)
            return np.column_stack([1 - probability, probability])

        def predict(self, values):
            return np.full(len(values), 0.5)

    monkeypatch.setattr(olist_evaluate, "EVIDENCE", tmp_path)
    monkeypatch.setattr(pd, "read_parquet", lambda path: frame.copy())
    monkeypatch.setattr(
        olist_evaluate.joblib, "load", lambda path: ("platt", Calibrator(), Calibrator())
    )
    written = {}
    monkeypatch.setattr(
        olist_evaluate, "write_table", lambda name, table: written.__setitem__(name, table.copy())
    )
    result = olist_evaluate.run(resamples=4)
    reference_rng = np.random.default_rng(olist_evaluate.SEED)
    for _ in range(4):
        reference_rng.integers(0, n, n)
    weeks = pd.to_datetime(frame["first_contact_date"], utc=True).dt.to_period("W").astype(str)
    unique_weeks = weeks.unique()
    values, labels = frame["value_90d"].to_numpy(float), frame["converted"].to_numpy()
    differences = []
    for _ in range(1000):
        selected = reference_rng.choice(unique_weeks, len(unique_weeks), replace=True)
        positions = np.concatenate([np.flatnonzero(weeks.to_numpy() == week) for week in selected])
        challenger = capacity_metrics(
            frame["accountpulse_ev_score"].to_numpy()[positions],
            values[positions],
            labels[positions],
            0.1,
        )
        baseline = capacity_metrics(
            frame["xgboost_score"].to_numpy()[positions], values[positions], labels[positions], 0.1
        )
        differences.append(challenger.value_capture - baseline.value_capture)
    assert result["apev_minus_xgboost_value_capture_10"]["temporal_block_ci_95"] == [
        float(np.quantile(differences, 0.025)),
        float(np.quantile(differences, 0.975)),
    ]
    assert len(written["bootstrap_results"]) == 8
