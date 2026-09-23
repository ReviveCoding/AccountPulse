"""Track-A source qualification, protected splits, labels, and PIT features."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import yaml

from accountpulse.evidence import EVIDENCE, write_table
from accountpulse.paths import REPO
from accountpulse.provenance import sha256

PROTOCOL = "AP-V1-TRACKA-20260923-A1"
ROOT = REPO / "data/incoming/maven"
CONFIG_PATH = REPO / "configs/track_a.yaml"
OBSERVATION_CUTOFF = pd.Timestamp("2017-12-31")
HORIZON = 60
PRODUCT_FIX = {"GTXPro": "GTX Pro"}
SPLIT_RANGES = {
    "FIT": (pd.Timestamp("2016-10-20"), pd.Timestamp("2017-03-19")),
    "VALIDATION": (pd.Timestamp("2017-05-19"), pd.Timestamp("2017-06-05")),
    "POLICY": (pd.Timestamp("2017-08-05"), pd.Timestamp("2017-08-14")),
    "LOCKED": (pd.Timestamp("2017-10-14"), pd.Timestamp("2017-11-01")),
}
GAP_RANGES = {
    "PURGE_1": (pd.Timestamp("2017-03-20"), pd.Timestamp("2017-05-18")),
    "PURGE_2": (pd.Timestamp("2017-06-06"), pd.Timestamp("2017-08-04")),
    "PURGE_3": (pd.Timestamp("2017-08-15"), pd.Timestamp("2017-10-13")),
}

EXPECTED_COLUMNS = {
    "sales_pipeline.csv": {
        "opportunity_id",
        "sales_agent",
        "product",
        "account",
        "deal_stage",
        "engage_date",
        "close_date",
        "close_value",
    },
    "accounts.csv": {
        "account",
        "sector",
        "year_established",
        "revenue",
        "employees",
        "office_location",
        "subsidiary_of",
    },
    "products.csv": {"product", "series", "sales_price"},
    "sales_teams.csv": {"sales_agent", "manager", "regional_office"},
    "data_dictionary.csv": {"Table", "Field", "Description"},
}

PREASSIGN_CATEGORICAL = [
    "sector",
    "office_location",
    "product",
    "series",
    "engage_month",
    "engage_quarter",
    "company_size",
    "account_missing",
]
PREASSIGN_NUMERIC = [
    "revenue",
    "employees",
    "company_age",
    "sales_price",
    "account_prior_opportunities",
    "account_prior_resolved",
    "account_prior_wins",
    "account_prior_win_rate",
    "account_prior_realized_value",
    "account_prior_avg_won_value",
    "account_prior_avg_terminal_days",
    "days_since_prior_opportunity",
    "product_prior_opportunities",
    "product_prior_resolved",
    "product_prior_win_rate",
    "product_prior_avg_won_value",
    "sector_prior_resolved",
    "sector_prior_win_rate",
    "sector_prior_avg_won_value",
]
POSTASSIGN_CATEGORICAL = ["sales_agent", "manager", "regional_office"]


def config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _split_for_date(value: pd.Timestamp | pd.NaT) -> str:
    if pd.isna(value):
        return "INELIGIBLE_NO_ENGAGE_DATE"
    day = pd.Timestamp(value).normalize()
    for name, (start, end) in SPLIT_RANGES.items():
        if start <= day <= end:
            return name
    for name, (start, end) in GAP_RANGES.items():
        if start <= day <= end:
            return name
    return "OUT_OF_PROTOCOL"


def _source_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pipeline = pd.read_csv(ROOT / "sales_pipeline.csv")
    accounts = pd.read_csv(ROOT / "accounts.csv")
    products = pd.read_csv(ROOT / "products.csv")
    teams = pd.read_csv(ROOT / "sales_teams.csv")
    pipeline["engage_date"] = pd.to_datetime(pipeline["engage_date"], errors="coerce")
    pipeline["close_date"] = pd.to_datetime(pipeline["close_date"], errors="coerce")
    pipeline["product_raw"] = pipeline["product"]
    pipeline["product"] = pipeline["product"].replace(PRODUCT_FIX)
    return pipeline, accounts, products, teams


def qualify_source() -> dict[str, Any]:
    cfg = config()
    hashes: dict[str, str] = {}
    shapes: dict[str, list[int]] = {}
    for filename, expected_hash in cfg["source_hashes"].items():
        path = ROOT / filename
        observed = sha256(path)
        if observed != expected_hash:
            raise ValueError(
                "SOURCE_INTEGRITY_FAILURE: "
                f"{filename} expected {expected_hash}, observed {observed}"
            )
        frame = pd.read_csv(path)
        if set(frame.columns) != EXPECTED_COLUMNS[filename]:
            raise ValueError(f"Schema mismatch for {filename}: {list(frame.columns)}")
        hashes[filename] = observed
        shapes[filename] = [len(frame), len(frame.columns)]

    pipeline, accounts, products, teams = _source_frames()
    terminal = pipeline["deal_stage"].isin(["Won", "Lost"])
    violations = {
        "duplicate_opportunity_id": int(pipeline["opportunity_id"].duplicated().sum()),
        "duplicate_accounts": int(accounts["account"].duplicated().sum()),
        "duplicate_products": int(products["product"].duplicated().sum()),
        "duplicate_agents": int(teams["sales_agent"].duplicated().sum()),
        "unknown_stage": int(
            (~pipeline["deal_stage"].isin(["Won", "Lost", "Engaging", "Prospecting"])).sum()
        ),
        "terminal_missing_close_date": int((terminal & pipeline["close_date"].isna()).sum()),
        "terminal_missing_close_value": int((terminal & pipeline["close_value"].isna()).sum()),
        "nonterminal_has_close_date": int((~terminal & pipeline["close_date"].notna()).sum()),
        "nonterminal_has_close_value": int((~terminal & pipeline["close_value"].notna()).sum()),
        "close_before_engage": int(
            (pipeline["close_date"] < pipeline["engage_date"]).fillna(False).sum()
        ),
        "negative_close_value": int((pipeline["close_value"] < 0).fillna(False).sum()),
        "lost_nonzero_value": int(
            ((pipeline["deal_stage"] == "Lost") & (pipeline["close_value"] != 0)).sum()
        ),
        "won_nonpositive_value": int(
            ((pipeline["deal_stage"] == "Won") & (pipeline["close_value"] <= 0)).sum()
        ),
        "orphan_account": int((~pipeline["account"].dropna().isin(accounts["account"])).sum()),
        "orphan_product_after_canonicalization": int(
            (~pipeline["product"].isin(products["product"])).sum()
        ),
        "orphan_agent": int((~pipeline["sales_agent"].isin(teams["sales_agent"])).sum()),
    }
    if any(violations.values()):
        raise ValueError(f"G1 schema qualification failed: {violations}")
    if len(pipeline) != 8800 or not pipeline["opportunity_id"].is_unique:
        raise ValueError("G0/G1 failed: expected 8,800 unique opportunities")

    stage_rows = [
        {"deal_stage": stage, "row_count": int(count)}
        for stage, count in pipeline["deal_stage"].value_counts().items()
    ]
    write_table("maven_stage_audit", pd.DataFrame(stage_rows))
    audit = {
        "protocol_id": PROTOCOL,
        "status": "QUALIFIED",
        "source_hashes": hashes,
        "shapes": shapes,
        "violations": violations,
        "missingness": {
            column: int(count) for column, count in pipeline.isna().sum().items() if int(count) > 0
        },
        "canonicalization": PRODUCT_FIX,
        "stage_counts": pipeline["deal_stage"].value_counts().to_dict(),
        "date_support": {
            "engage_min": str(pipeline["engage_date"].min().date()),
            "engage_max": str(pipeline["engage_date"].max().date()),
            "close_min": str(pipeline["close_date"].min().date()),
            "close_max": str(pipeline["close_date"].max().date()),
        },
        "claim_boundary": "FICTITIOUS_PUBLIC_CRM_NOT_PRODUCTION",
    }
    (EVIDENCE / "maven_source_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def qualify_horizon_and_splits() -> dict[str, Any]:
    pipeline, _, _, _ = _source_frames()
    dates = pipeline["engage_date"].dropna()
    minima = config()["horizon_qualification_minimum_rows"]
    candidates = []
    fixed_ranges = {
        90: None,
        60: config()["splits"],
        30: {
            "FIT": ["2016-10-20", "2017-05-20"],
            "VALIDATION": ["2017-06-20", "2017-07-17"],
            "POLICY": ["2017-08-17", "2017-09-08"],
            "LOCKED": ["2017-10-09", "2017-12-01"],
        },
    }
    for horizon in (90, 60, 30):
        ranges = fixed_ranges[horizon]
        if ranges is None:
            counts = {"FIT": 1, "VALIDATION": 13, "POLICY": 35, "LOCKED": 2233}
        else:
            counts = {
                name: int(dates.between(pd.Timestamp(start), pd.Timestamp(end)).sum())
                for name, (start, end) in ranges.items()
            }
        passes = all(counts[name] >= minima[name] for name in minima)
        candidates.append(
            {
                "horizon_days": horizon,
                "partition_counts": counts,
                "minimum_counts": minima,
                "passes_timestamp_and_row_count_support": passes,
            }
        )
    selected = next(item for item in candidates if item["passes_timestamp_and_row_count_support"])
    if selected["horizon_days"] != HORIZON:
        raise ValueError("Configured horizon does not match largest defensible candidate")
    result = {
        "protocol_id": PROTOCOL,
        "selection_rule": (
            "largest candidate in [90,60,30] supporting 60-day purge gaps and preregistered "
            "minimum partition row counts; outcomes not used"
        ),
        "candidates": candidates,
        "selected_horizon_days": HORIZON,
        "observation_cutoff": str(OBSERVATION_CUTOFF.date()),
    }
    (EVIDENCE / "maven_horizon_qualification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _empty_stats() -> dict[str, float]:
    return {
        "resolved": 0.0,
        "wins": 0.0,
        "won_value": 0.0,
        "terminal_days": 0.0,
    }


def build_features() -> dict[str, Any]:
    """Build pre-lock features; LOCKED outcomes never update historical aggregates."""
    pipeline, accounts, products, teams = _source_frames()
    pipeline["split"] = pipeline["engage_date"].map(_split_for_date)
    frame = (
        pipeline.merge(accounts, on="account", how="left", validate="m:1")
        .merge(products, on="product", how="left", validate="m:1")
        .merge(teams, on="sales_agent", how="left", validate="m:1")
    )
    frame["sector"] = frame["sector"].fillna("__UNKNOWN__")
    frame["office_location"] = frame["office_location"].fillna("__UNKNOWN__")
    frame["account_key"] = frame["account"].fillna("__MISSING_ACCOUNT__")
    frame["account_missing"] = frame["account"].isna().map({True: "YES", False: "NO"})
    frame["engage_month"] = frame["engage_date"].dt.month.astype("Int64").astype("string")
    frame["engage_quarter"] = frame["engage_date"].dt.quarter.astype("Int64").astype("string")
    frame["company_age"] = frame["engage_date"].dt.year - frame["year_established"]
    frame["company_size"] = (
        pd.cut(
            frame["employees"],
            bins=[-np.inf, 100, 500, 2000, np.inf],
            labels=["SMALL", "MEDIUM", "LARGE", "ENTERPRISE"],
        )
        .astype("string")
        .fillna("__UNKNOWN__")
    )
    frame["terminal_days"] = (frame["close_date"] - frame["engage_date"]).dt.days

    indexed = frame.sort_values(["engage_date", "opportunity_id"], na_position="last")
    account_engagements: defaultdict[str, int] = defaultdict(int)
    product_engagements: defaultdict[str, int] = defaultdict(int)
    last_account_date: dict[str, pd.Timestamp] = {}
    account_stats: defaultdict[str, dict[str, float]] = defaultdict(_empty_stats)
    product_stats: defaultdict[str, dict[str, float]] = defaultdict(_empty_stats)
    sector_stats: defaultdict[str, dict[str, float]] = defaultdict(_empty_stats)
    pending: list[tuple[pd.Timestamp, str, str, str, str, float, float]] = []
    feature_rows: list[dict[str, Any]] = []

    eligible = indexed[indexed["engage_date"].notna()]
    for engage_date, batch in eligible.groupby("engage_date", sort=True):
        score_time = pd.Timestamp(engage_date)
        while pending and pending[0][0] < score_time:
            close_date, _, account_key, product, sector, won, value, terminal_days = heapq.heappop(
                pending
            )
            del close_date
            for key, stats_map in (
                (account_key, account_stats),
                (product, product_stats),
                (sector, sector_stats),
            ):
                stats = stats_map[key]
                stats["resolved"] += 1
                stats["wins"] += won
                stats["won_value"] += value
                stats["terminal_days"] += terminal_days

        for row in batch.itertuples(index=False):
            account_key = str(row.account_key)
            product = str(row.product)
            sector = str(row.sector)
            a = account_stats[account_key]
            p = product_stats[product]
            s = sector_stats[sector]
            prior_date = last_account_date.get(account_key)
            feature_rows.append(
                {
                    "opportunity_id": row.opportunity_id,
                    "split": row.split,
                    "engage_date": row.engage_date,
                    "account": row.account,
                    "account_key": account_key,
                    "sales_agent": row.sales_agent,
                    "manager": row.manager,
                    "regional_office": row.regional_office,
                    "sector": sector,
                    "office_location": row.office_location,
                    "product": product,
                    "series": row.series,
                    "engage_month": row.engage_month,
                    "engage_quarter": row.engage_quarter,
                    "company_size": row.company_size,
                    "account_missing": row.account_missing,
                    "revenue": row.revenue,
                    "employees": row.employees,
                    "company_age": row.company_age,
                    "sales_price": row.sales_price,
                    "account_prior_opportunities": account_engagements[account_key],
                    "account_prior_resolved": a["resolved"],
                    "account_prior_wins": a["wins"],
                    "account_prior_win_rate": (a["wins"] + 1) / (a["resolved"] + 2),
                    "account_prior_realized_value": a["won_value"],
                    "account_prior_avg_won_value": a["won_value"] / max(a["wins"], 1),
                    "account_prior_avg_terminal_days": a["terminal_days"] / max(a["resolved"], 1),
                    "days_since_prior_opportunity": (
                        (score_time - prior_date).days if prior_date is not None else np.nan
                    ),
                    "product_prior_opportunities": product_engagements[product],
                    "product_prior_resolved": p["resolved"],
                    "product_prior_win_rate": (p["wins"] + 1) / (p["resolved"] + 2),
                    "product_prior_avg_won_value": p["won_value"] / max(p["wins"], 1),
                    "sector_prior_resolved": s["resolved"],
                    "sector_prior_win_rate": (s["wins"] + 1) / (s["resolved"] + 2),
                    "sector_prior_avg_won_value": s["won_value"] / max(s["wins"], 1),
                }
            )

        for row in batch.itertuples(index=False):
            account_key = str(row.account_key)
            account_engagements[account_key] += 1
            product_engagements[str(row.product)] += 1
            last_account_date[account_key] = score_time
            terminal = row.deal_stage in ("Won", "Lost") and pd.notna(row.close_date)
            # Protected LOCKED outcomes cannot enter any pre-lock feature, even for later rows.
            if terminal and row.split != "LOCKED":
                won = float(row.deal_stage == "Won")
                value = float(row.close_value) if won else 0.0
                heapq.heappush(
                    pending,
                    (
                        pd.Timestamp(row.close_date),
                        str(row.opportunity_id),
                        account_key,
                        str(row.product),
                        str(row.sector),
                        won,
                        value,
                        float(row.terminal_days),
                    ),
                )

    features = pd.DataFrame(feature_rows).sort_values(["engage_date", "opportunity_id"])
    if {"sales_agent", "manager", "regional_office"} & set(
        PREASSIGN_CATEGORICAL + PREASSIGN_NUMERIC
    ):
        raise ValueError("PREASSIGN contract contains assignment fields")
    if not np.isfinite(features[PREASSIGN_NUMERIC].fillna(0).to_numpy(dtype=float)).all():
        raise ValueError("Non-finite Track-A feature")

    identity = features[["opportunity_id", "split", "engage_date", "account_key"]].copy()
    write_table("maven_split_identity", identity)
    locked = features[features["split"] == "LOCKED"].copy()
    write_table("maven_locked_features", locked)

    allowed_label_splits = {
        "FIT",
        "VALIDATION",
        "POLICY",
        "PURGE_1",
        "PURGE_2",
        "PURGE_3",
    }
    development_ids = set(
        features.loc[features["split"].isin(allowed_label_splits), "opportunity_id"]
    )
    raw_development = frame[frame["opportunity_id"].isin(development_ids)].copy()
    raw_development["event_within_horizon"] = raw_development["deal_stage"].isin(
        ["Won", "Lost"]
    ) & raw_development["terminal_days"].le(HORIZON)
    raw_development["won"] = (
        (raw_development["deal_stage"] == "Won") & raw_development["terminal_days"].le(HORIZON)
    ).astype(int)
    raw_development["lost"] = (
        (raw_development["deal_stage"] == "Lost") & raw_development["terminal_days"].le(HORIZON)
    ).astype(int)
    raw_development["realized_value"] = np.where(
        raw_development["won"].eq(1), raw_development["close_value"], 0.0
    )
    raw_development["event_time"] = np.where(
        raw_development["event_within_horizon"],
        raw_development["terminal_days"],
        HORIZON,
    )
    raw_development["terminal_state"] = np.select(
        [raw_development["won"].eq(1), raw_development["lost"].eq(1)],
        ["WON", "LOST"],
        default="OPEN_CENSORED_60D",
    )
    raw_development["time_bin"] = np.select(
        [
            raw_development["event_within_horizon"] & raw_development["terminal_days"].le(30),
            raw_development["event_within_horizon"] & raw_development["terminal_days"].gt(30),
        ],
        [1, 2],
        default=0,
    ).astype(int)
    labels = raw_development[
        [
            "opportunity_id",
            "won",
            "lost",
            "realized_value",
            "event_time",
            "event_within_horizon",
            "terminal_state",
            "time_bin",
        ]
    ]
    development = features[features["opportunity_id"].isin(development_ids)].merge(
        labels, on="opportunity_id", how="left", validate="1:1"
    )
    write_table("maven_development", development)

    split_counts = features["split"].value_counts().to_dict()
    selected_counts = {name: int(split_counts.get(name, 0)) for name in config()["splits"]}
    expected = {"FIT": 1670, "VALIDATION": 434, "POLICY": 273, "LOCKED": 403}
    if selected_counts != expected:
        raise ValueError(f"Split identity mismatch: {selected_counts} != {expected}")
    split_hash = hashlib.sha256(
        identity.sort_values("opportunity_id").to_csv(index=False).encode()
    ).hexdigest()
    result = {
        "protocol_id": PROTOCOL,
        "horizon_days": HORIZON,
        "split_counts": split_counts,
        "selected_split_counts": selected_counts,
        "split_identity_sha256": split_hash,
        "preassign_features": PREASSIGN_CATEGORICAL + PREASSIGN_NUMERIC,
        "postassign_additions": POSTASSIGN_CATEGORICAL,
        "locked_labels_materialized": False,
        "pit_rule": "resolved outcomes update aggregates only when close_date < engage_date",
        "locked_history_rule": "LOCKED outcomes never update pre-lock feature aggregates",
        "generated_at": datetime.now(UTC).isoformat(),
    }
    (EVIDENCE / "maven_feature_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def locked_outcomes() -> pd.DataFrame:
    """Read protected labels only after the freeze guard has authorized access."""
    from accountpulse.protection import assert_locked_access

    assert_locked_access()
    identity = pd.read_parquet(EVIDENCE / "maven_split_identity.parquet")
    locked_ids = set(identity.loc[identity["split"] == "LOCKED", "opportunity_id"])
    pipeline, _, _, _ = _source_frames()
    locked = pipeline[pipeline["opportunity_id"].isin(locked_ids)].copy()
    locked["terminal_days"] = (locked["close_date"] - locked["engage_date"]).dt.days
    locked["event_within_horizon"] = locked["deal_stage"].isin(["Won", "Lost"]) & locked[
        "terminal_days"
    ].le(HORIZON)
    locked["won"] = ((locked["deal_stage"] == "Won") & locked["terminal_days"].le(HORIZON)).astype(
        int
    )
    locked["lost"] = (
        (locked["deal_stage"] == "Lost") & locked["terminal_days"].le(HORIZON)
    ).astype(int)
    locked["realized_value"] = np.where(locked["won"].eq(1), locked["close_value"], 0.0)
    locked["event_time"] = np.where(
        locked["event_within_horizon"], locked["terminal_days"], HORIZON
    )
    locked["terminal_state"] = np.select(
        [locked["won"].eq(1), locked["lost"].eq(1)],
        ["WON", "LOST"],
        default="OPEN_CENSORED_60D",
    )
    locked["time_bin"] = np.select(
        [
            locked["event_within_horizon"] & locked["terminal_days"].le(30),
            locked["event_within_horizon"] & locked["terminal_days"].gt(30),
        ],
        [1, 2],
        default=0,
    ).astype(int)
    if len(locked) != 403:
        raise ValueError("LOCKED identity count changed")
    return locked[
        [
            "opportunity_id",
            "won",
            "lost",
            "realized_value",
            "event_time",
            "event_within_horizon",
            "terminal_state",
            "time_bin",
        ]
    ]
