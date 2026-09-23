# ruff: noqa: E501
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from accountpulse.paths import REPO
from accountpulse.tracka_diagnostics import (
    BASE_PROTOCOL,
    DIAGNOSTIC_PROTOCOL,
    _rank_stability,
    _selected,
    capacity_is_monotone,
    hhi,
    protected_hashes,
)


def test_local_selection_differs_from_global_slice_membership() -> None:
    scores = np.arange(100, dtype=float)
    global_selected = _selected(scores)
    local_selected = _selected(scores[:20])
    assert global_selected[:20].sum() == 0
    assert local_selected.sum() == 2


def test_model_disagreement_set_math() -> None:
    left = np.array([True, True, False, False])
    right = np.array([True, False, True, False])
    assert int((left & right).sum()) == 1
    assert int((left | right).sum()) == 3
    assert (left & right).sum() / (left | right).sum() == 1 / 3


def test_hhi() -> None:
    assert hhi([1, 1, 1, 1]) == 0.25
    assert hhi([4, 0, 0, 0]) == 1.0


def test_capacity_curve_monotonicity() -> None:
    curve = pd.DataFrame({"model": ["A", "A", "B", "B"], "capacity": [0.1, 0.2, 0.1, 0.2], "value_capture": [0.2, 0.3, 0.4, 0.4]})
    assert capacity_is_monotone(curve)
    curve.loc[1, "value_capture"] = 0.1
    assert not capacity_is_monotone(curve)


def test_bootstrap_stability_probabilities_and_support() -> None:
    frame = pd.DataFrame({"opportunity_id": [f"o{i}" for i in range(20)], "account_key": [f"a{i // 2}" for i in range(20)], "B1_business_heuristic": np.arange(20), "AccountPulse_EV": np.arange(20)[::-1], "LambdaMART": np.linspace(0, 1, 20)})
    result = _rank_stability(frame, replicates=20)
    opportunities = result[result["record_type"] == "OPPORTUNITY_RANK"]
    assert len(opportunities) == 60
    assert opportunities[["p_top5", "p_top10", "p_top20"]].stack().between(0, 1).all()


def test_diagnostic_manifest_and_frozen_immutability() -> None:
    manifest = yaml.safe_load((REPO / "DIAGNOSTICS_MANIFEST.yaml").read_text(encoding="utf-8"))
    assert manifest["base_protocol"] == BASE_PROTOCOL
    assert manifest["diagnostic_protocol"] == DIAGNOSTIC_PROTOCOL
    assert manifest["post_lock"] is True
    assert manifest["decision_eligible"] is False
    assert manifest["protected_hashes_before"] == protected_hashes()
    assert manifest["protected_hashes_after"] == protected_hashes()


def test_slice_support_threshold_is_enforced_in_intersection_artifact() -> None:
    artifact = pd.read_parquet(REPO / "artifacts/evidence/maven_intersection_slice_metrics_postlock.parquet")
    assert artifact["support_n"].dropna().ge(30).all()
