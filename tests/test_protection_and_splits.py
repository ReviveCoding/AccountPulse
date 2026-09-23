from __future__ import annotations

import pandas as pd
import pytest

from accountpulse.gpu import GPURequiredError, require_cuda
from accountpulse.protection import ProtectedAccessError, assert_locked_access, refuse_force


def test_locked_access_fails_closed() -> None:
    with pytest.raises(ProtectedAccessError, match="not FROZEN"):
        assert_locked_access()


def test_force_cannot_bypass() -> None:
    with pytest.raises(ProtectedAccessError, match="cannot bypass"):
        refuse_force(True)


def test_gpu_required_env_cannot_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCOUNT_PULSE_REQUIRE_CUDA", "0")
    with pytest.raises(GPURequiredError):
        require_cuda()


def test_olist_label_embargo() -> None:
    frame = pd.read_parquet("artifacts/evidence/olist_leads.parquet")
    fit_max = frame.loc[frame["split"] == "FIT", "first_contact_date"].max()
    validation_min = frame.loc[frame["split"] == "VALIDATION", "first_contact_date"].min()
    validation_max = frame.loc[frame["split"] == "VALIDATION", "first_contact_date"].max()
    test_min = frame.loc[frame["split"] == "TEST", "first_contact_date"].min()
    assert fit_max + pd.Timedelta(days=90) < validation_min
    assert validation_max + pd.Timedelta(days=90) <= test_min


def test_uci_label_embargo_and_cold_definition() -> None:
    frame = pd.read_parquet("artifacts/evidence/uci_lifecycle_snapshots.parquet")
    fit_max = frame.loc[frame["split"] == "FIT", "cutoff"].max()
    validation = frame.loc[frame["split"] == "VALIDATION", "cutoff"].iloc[0]
    test = frame.loc[frame["split"] == "TEST", "cutoff"].iloc[0]
    assert fit_max + pd.Timedelta(days=90) <= validation
    assert validation + pd.Timedelta(days=90) <= test
    test_rows = frame[frame["split"] == "TEST"]
    assert (test_rows.loc[test_rows["customer_cohort"] == "UNSEEN", "first_seen"] > fit_max).all()
