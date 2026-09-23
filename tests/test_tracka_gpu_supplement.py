from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from accountpulse.paths import REPO
from accountpulse.tracka_gpu_supplement import (
    EXPECTED_PROTECTED,
    empirical_grid,
    map_encoded_feature,
    protected_hashes,
    validate_shap_additivity,
    value_capture_10,
)


def test_gpu_shap_additivity_validator() -> None:
    contributions = np.array([[0.2, -0.1, 0.5], [0.1, 0.3, -0.2]])
    margins = np.array([0.6, 0.2])
    assert validate_shap_additivity(contributions, margins) < 1e-12


def test_feature_name_mapping() -> None:
    assert map_encoded_feature("product_GTX Basic") == ("product", "product")
    assert map_encoded_feature("account_prior_win_rate") == (
        "account_prior_win_rate",
        "account_history",
    )
    assert map_encoded_feature("sector_prior_win_rate") == (
        "sector_prior_win_rate",
        "sector_history",
    )
    assert map_encoded_feature("revenue") == ("revenue", "firmographics")


def test_grouped_permutation_scoring_is_deterministic() -> None:
    scores = np.array([4.0, 3.0, 2.0, 1.0])
    values = np.array([10.0, 0.0, 5.0, 0.0])
    identifiers = pd.Series(["a", "b", "c", "d"])
    first = value_capture_10(scores, values, identifiers)
    second = value_capture_10(scores, values, identifiers)
    assert first == second == 2 / 3


def test_pdp_grid_stays_within_empirical_bounds() -> None:
    series = pd.Series(np.arange(101, dtype=float))
    grid = empirical_grid(series)
    assert grid[0] == series.quantile(0.05)
    assert grid[-1] == series.quantile(0.95)
    assert np.all(np.diff(grid) > 0)


def test_apev_component_reconstruction_artifact() -> None:
    artifact = pd.read_parquet(
        REPO / "artifacts/evidence/maven_apev_components_gpu_postlock.parquet"
    )
    detail = artifact[artifact["record_type"].eq("OPPORTUNITY_TIME_BIN")]
    assert detail["opportunity_id"].nunique() == 403
    assert detail.groupby("opportunity_id").size().eq(2).all()
    assert detail["reconstruction_error"].abs().max() < 1e-4
    assert detail["rho"].dropna().eq(0.005).all()

    opportunity = artifact[artifact["record_type"].eq("OPPORTUNITY_DISCOUNT_SUMMARY")]
    assert opportunity["opportunity_id"].nunique() == 403
    normalized = [
        "absolute_discount_penalty",
        "relative_discount_penalty",
        "effective_discount_factor",
        "expected_component_days",
        "conditional_expected_value",
    ]
    assert opportunity[normalized].notna().all().all()
    assert np.allclose(
        opportunity["relative_discount_penalty"]
        + opportunity["effective_discount_factor"],
        1.0,
    )
    relationships = artifact[artifact["record_type"].eq("DISCOUNT_RELATIONSHIP")]
    assert set(relationships["penalty_measure"]) == {
        "absolute_discount_penalty",
        "relative_discount_penalty",
    }
    assert set(relationships["summary_dimension"]) == {
        "sales_price",
        "realized_value",
        "expected_component_days",
        "undiscounted_total",
        "conditional_expected_value",
    }


def test_permutation_intervals_are_labeled_as_perturbation_quantiles() -> None:
    artifact = pd.read_parquet(
        REPO / "artifacts/evidence/maven_grouped_permutation_gpu_postlock.parquet"
    )
    assert {"permutation_q025", "permutation_q975", "permutation_distribution_interval_95"} <= set(
        artifact.columns
    )
    assert "ci_2_5" not in artifact.columns
    assert "ci_97_5" not in artifact.columns
    assert artifact["uncertainty_scope"].str.contains("NOT_SAMPLING_CONFIDENCE_INTERVAL").all()


def test_gpu_supplement_manifest_and_protected_hashes() -> None:
    manifest = yaml.safe_load(
        (REPO / "DIAGNOSTICS_GPU_SUPPLEMENT_MANIFEST.yaml").read_text(encoding="utf-8")
    )
    assert manifest["protected_byte_identical"] is True
    assert manifest["diag1_initial_gpu_status_preserved"] is True
    assert manifest["a1_decision"] == "RETAIN_BASELINE"
    assert manifest["protected_hashes_before"] == EXPECTED_PROTECTED
    assert manifest["protected_hashes_after"] == protected_hashes()


def test_no_forbidden_scientific_entry_points_in_gpu_runner() -> None:
    source = (REPO / "src/accountpulse/tracka_gpu_supplement.py").read_text(encoding="utf-8")
    assert "locked_" + "evaluate(" not in source
    assert "run_" + "development(" not in source
    assert "freeze(" not in source
