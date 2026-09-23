from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import accountpulse.evidence as evidence
import accountpulse.provenance as provenance
from accountpulse.causal import (
    _policy_metrics_from_signal,
    aipw_signal,
    deterministic_order,
    hash_split,
)
from accountpulse.olist import _break_ties
from accountpulse.paths import _repository_root
from accountpulse.reporting import _validate_report


def test_criteo_source_hash_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "criteo.csv.gz"
    source.write_bytes(b"verified fixture")
    expected = provenance.sha256(source)
    monkeypatch.setattr(provenance, "CRITEO_EXPECTED_SHA256", expected)
    assert provenance.validate_criteo_source(source) == expected
    source.write_bytes(b"tampered fixture")
    with pytest.raises(ValueError, match="SOURCE_INTEGRITY_FAILURE"):
        provenance.validate_criteo_source(source)


def test_track_d_hash_split_is_deterministic_and_disjoint() -> None:
    row_ids = np.arange(50_000, dtype=np.uint64)
    first = hash_split(row_ids)
    second = hash_split(row_ids)
    assert np.array_equal(first, second)
    assert set(first) == {"TRAIN", "VALIDATION", "TEST"}
    members = {name: set(row_ids[first == name]) for name in set(first)}
    assert members["TRAIN"].isdisjoint(members["VALIDATION"])
    assert members["TRAIN"].isdisjoint(members["TEST"])
    assert members["VALIDATION"].isdisjoint(members["TEST"])
    assert sum(map(len, members.values())) == len(row_ids)


def test_tie_breaking_is_deterministic_and_outcome_blind() -> None:
    identifiers = pd.Series(["c", "a", "d", "b"])
    scores = np.ones(4)
    first = _break_ties(scores, identifiers)
    second = _break_ties(scores, identifiers)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 4
    order = deterministic_order(np.ones(4), np.arange(4, dtype=np.uint64))
    assert np.array_equal(order, deterministic_order(np.ones(4), np.arange(4)))


def test_aipw_and_uplift_policy_recover_heterogeneous_effect() -> None:
    rng = np.random.default_rng(20260922)
    n = 200_000
    row_ids = np.arange(n, dtype=np.uint64)
    segment = rng.integers(0, 2, size=n)
    propensity = np.where(segment == 1, 0.8, 0.6)
    treatment = rng.binomial(1, propensity)
    mu0 = np.where(segment == 1, 0.05, 0.40)
    mu1 = np.where(segment == 1, 0.30, 0.42)
    outcome = rng.binomial(1, np.where(treatment == 1, mu1, mu0))
    signal = aipw_signal(treatment, outcome, propensity, mu0, mu1)
    assert signal.mean() == pytest.approx(0.135, abs=0.006)
    response_rows, _ = _policy_metrics_from_signal(
        mu1, treatment, outcome, row_ids, [0.5], signal
    )
    uplift_rows, _ = _policy_metrics_from_signal(
        mu1 - mu0, treatment, outcome, row_ids, [0.5], signal
    )
    assert uplift_rows[0]["aipw_incremental_per_eligible"] > response_rows[0][
        "aipw_incremental_per_eligible"
    ]


def test_evidence_upsert_is_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(evidence, "EVIDENCE", tmp_path)
    monkeypatch.setattr(evidence, "WAREHOUSE", tmp_path / "warehouse.duckdb")
    first = pd.DataFrame([{"protocol": "p", "stage": "s", "value": 1.0}])
    second = pd.DataFrame([{"protocol": "p", "stage": "s", "value": 2.0}])
    evidence.upsert_table("runs", first, ["protocol", "stage"])
    evidence.upsert_table("runs", second, ["protocol", "stage"])
    result = pd.read_parquet(tmp_path / "runs.parquet")
    assert result.to_dict(orient="records") == [
        {"protocol": "p", "stage": "s", "value": 2.0}
    ]


def test_repository_root_prefers_explicit_then_working_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    (project / "artifacts/state").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (project / "artifacts/state/pipeline_state.json").write_text("{}", encoding="utf-8")
    monkeypatch.delenv("ACCOUNT_PULSE_REPO", raising=False)
    monkeypatch.chdir(project)
    assert _repository_root() == project.resolve()
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("ACCOUNT_PULSE_REPO", str(explicit))
    assert _repository_root() == explicit.resolve()


def test_current_criteo_manifest_and_claim_boundary() -> None:
    manifest = yaml.safe_load(Path("SOURCE_MANIFEST.yaml").read_text(encoding="utf-8"))
    source = next(
        item for item in manifest["sources"] if item["source_name"] == "criteo_uplift_v2_1"
    )
    assert source["version"].endswith(provenance.CRITEO_REVISION)
    assert source["files"][0]["sha256"] == provenance.CRITEO_EXPECTED_SHA256
    assert "NOT_TRANSPORTABILITY" in source["claim_boundary"]
    result = json.loads(Path("artifacts/evidence/causal_result.json").read_text())
    assert result["protocol_id"].endswith("D2")
    assert "no transportability" in result["claim_boundary"].lower()


def test_reporting_rejects_stale_causal_blocker_claim(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    report.write_text("Track D is blocked", encoding="utf-8")
    with pytest.raises(ValueError, match="Stale Track-D blocker wording"):
        _validate_report(report)
    report.write_text("Track D completed as a separate causal bridge.", encoding="utf-8")
    _validate_report(report)
