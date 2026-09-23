from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from accountpulse.cli import score_maven_locked_replay
from accountpulse.tracka_data import PREASSIGN_CATEGORICAL, PREASSIGN_NUMERIC
from accountpulse.tracka_models import _tie_break

EXPECTED = {
    "sales_pipeline.csv": "825ce8f6c32d4009548b468df3173d55a46fd73f2531f532c5459371dc52adf2",
    "accounts.csv": "e5242324768a563fc632cddfed49a29acbbf2892b8a3c6453cc9650de9ae0358",
    "products.csv": "7c1c8cbbdb6d4c286902e1985eeb529a36366d6a43f43cd4a93c4b1da2a6eb84",
    "sales_teams.csv": "aeff1272ebe196f5a27e3fc0578aa27abf48ed9ae461aa344fb95990e5ad8bd1",
    "data_dictionary.csv": "22b34e498d07e3d7f322afdbf81d70a5dc0a389792944e50ca2af86a3597f0af",
}


def test_tracka_sources_are_exact_official_files() -> None:
    root = Path("data/incoming/maven")
    for name, expected in EXPECTED.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    assert len(pd.read_csv(root / "sales_pipeline.csv")) == 8800


def test_tracka_splits_are_disjoint_and_embargoed() -> None:
    frame = pd.read_parquet("artifacts/evidence/maven_split_identity.parquet")
    selected = frame[frame["split"].isin(["FIT", "VALIDATION", "POLICY", "LOCKED"])]
    assert selected["opportunity_id"].is_unique
    edges = {
        name: (
            selected.loc[selected["split"] == name, "engage_date"].min(),
            selected.loc[selected["split"] == name, "engage_date"].max(),
        )
        for name in ["FIT", "VALIDATION", "POLICY", "LOCKED"]
    }
    assert edges["FIT"][1] + pd.Timedelta(days=60) < edges["VALIDATION"][0]
    assert edges["VALIDATION"][1] + pd.Timedelta(days=60) < edges["POLICY"][0]
    assert edges["POLICY"][1] + pd.Timedelta(days=60) < edges["LOCKED"][0]


def test_preassign_contract_excludes_assignment_and_future_fields() -> None:
    features = set(PREASSIGN_CATEGORICAL + PREASSIGN_NUMERIC)
    assert not features & {
        "sales_agent",
        "manager",
        "regional_office",
        "deal_stage",
        "close_date",
        "close_value",
    }


def test_historical_counts_use_only_strictly_prior_resolutions() -> None:
    raw = pd.read_csv("data/incoming/maven/sales_pipeline.csv")
    raw["engage_date"] = pd.to_datetime(raw["engage_date"])
    raw["close_date"] = pd.to_datetime(raw["close_date"])
    features = pd.read_parquet("artifacts/evidence/maven_development.parquet")
    sample = features.loc[features["split"].isin(["VALIDATION", "POLICY"])].iloc[::53]
    for row in sample.itertuples():
        expected = raw[
            raw["account"].fillna("__MISSING_ACCOUNT__").eq(row.account_key)
            & raw["close_date"].lt(row.engage_date)
            & raw["deal_stage"].isin(["Won", "Lost"])
        ]
        assert row.account_prior_resolved == len(expected)


def test_tie_break_is_deterministic_and_id_sensitive() -> None:
    identifiers = pd.Series(["A", "B", "C"])
    first = _tie_break(np.ones(3), identifiers)
    second = _tie_break(np.ones(3), identifiers)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 3


def test_locked_decision_and_claim_boundary() -> None:
    result = json.loads(Path("artifacts/evidence/maven_locked_result.json").read_text())
    assert result["decision"] == "RETAIN_BASELINE"
    assert result["gates"]["G5_value_capture"] == "FAIL"
    assert result["gates"]["G9_cold_account_robustness"] == "NOT_EVALUABLE"
    manifest = yaml.safe_load(Path("FREEZE_MANIFEST.yaml").read_text())
    assert manifest["locked_outcomes_opened"] is False
    receipt = json.loads(Path("LOCKED_START_RECEIPT.json").read_text())
    assert (
        receipt["freeze_manifest_sha256"]
        == hashlib.sha256(Path("FREEZE_MANIFEST.yaml").read_bytes()).hexdigest()
    )
    assert "causal lift" in manifest["claim_templates"]["prohibited"]


def test_maven_batch_replay_uses_retained_model() -> None:
    scored = score_maven_locked_replay("2017-11-01", 0.10, None)
    assert len(scored) == 403
    assert scored["rank"].is_unique
    assert scored["selected"].sum() == 41
    assert (scored["model_version"] == "tracka-a1-business-heuristic-retained").all()


def test_postlock_secondary_metrics_are_finite_and_decision_independent() -> None:
    ranking = pd.read_parquet("artifacts/evidence/maven_ranking_secondary_locked.parquet")
    survival = pd.read_parquet("artifacts/evidence/maven_survival_metrics_locked.parquet")
    values = pd.read_parquet("artifacts/evidence/maven_value_metrics_locked.parquet")
    assert len(ranking) == 52
    assert np.isfinite(ranking[["precision_at_k", "recall_at_k", "ndcg"]]).all().all()
    assert np.isfinite(survival["value"]).all()
    assert np.isfinite(values[["mae", "rmse"]]).all().all()
    assert set(ranking["protocol_id"]) == {"AP-V1-TRACKA-20260923-A1"}
