"""Reusable EDA tables and publication-ready static figures."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from accountpulse.evidence import EVIDENCE, write_table
from accountpulse.paths import REPO


def run() -> dict[str, object]:
    figure_dir = REPO / "artifacts/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    olist = pd.read_parquet(EVIDENCE / "olist_leads.parquet")
    snapshots = pd.read_parquet(EVIDENCE / "uci_lifecycle_snapshots.parquet")
    baskets = pd.read_parquet(EVIDENCE / "uci_baskets.parquet")
    summaries = []
    for name, frame in (
        ("olist_leads", olist),
        ("uci_snapshots", snapshots),
        ("uci_baskets", baskets),
    ):
        summaries.append(
            {
                "dataset": name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "missing_cells": int(frame.isna().sum().sum()),
                "duplicate_rows": int(frame.duplicated().sum()),
            }
        )
    write_table("eda_summary", pd.DataFrame(summaries))

    origin = (
        olist.groupby("origin", dropna=False)
        .agg(leads=("mql_id", "size"), conversion=("converted", "mean"))
        .sort_values("leads")
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    origin["leads"].plot.barh(
        ax=axes[0], color="#355C7D", title="Olist leads by first-contact origin"
    )
    origin["conversion"].plot.barh(ax=axes[1], color="#F67280", title="Observed conversion rate")
    axes[1].set_xlim(0, max(0.3, float(origin["conversion"].max()) * 1.1))
    fig.tight_layout()
    fig.savefig(figure_dir / "olist_origin_conversion.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    positive_value = olist.loc[olist["value_90d"] > 0, "value_90d"]
    ax.hist(np.log1p(positive_value), bins=35, color="#6C5B7B")
    ax.set(
        title="Olist 90-day seller value (positive leads)", xlabel="log(1 + value)", ylabel="Leads"
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "olist_value_log_distribution.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(np.log1p(baskets["positive_spend"]), bins=60, color="#355C7D")
    axes[0].set(title="UCI basket spend", xlabel="log(1 + positive spend)")
    cadence = snapshots[snapshots["split"] == "TEST"]["recency_days"].clip(upper=365)
    axes[1].hist(cadence, bins=50, color="#99B898")
    axes[1].set(title="UCI test recency", xlabel="Days since prior basket")
    fig.tight_layout()
    fig.savefig(figure_dir / "uci_spend_recency.png", dpi=180)
    plt.close(fig)

    country = (
        baskets.groupby("country")
        .agg(baskets=("invoice", "size"), spend=("positive_spend", "sum"))
        .nlargest(12, "baskets")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    country["baskets"].sort_values().plot.barh(
        ax=ax, color="#C06C84", title="UCI baskets: leading countries"
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "uci_country_baskets.png", dpi=180)
    plt.close(fig)

    result = {
        "status": "COMPLETE",
        "figures": sorted(str(path.relative_to(REPO)) for path in figure_dir.glob("*.png")),
        "track_a": "SCHEMA_PREVIEW_ONLY_BLOCKED_PRIMARY",
        "track_b": "COMPLETE",
        "track_c": "COMPLETE",
    }
    (EVIDENCE / "eda_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
