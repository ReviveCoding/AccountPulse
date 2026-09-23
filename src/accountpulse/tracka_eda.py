"""Reusable Track-A descriptive evidence; never used for post-lock model tuning."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from accountpulse.evidence import write_table
from accountpulse.paths import REPO
from accountpulse.tracka_data import PROTOCOL, _source_frames


def run() -> dict[str, object]:
    pipeline, accounts, products, _teams = _source_frames()
    frame = pipeline.merge(accounts, on="account", how="left").merge(products, on="product")
    frame["terminal_days"] = (frame["close_date"] - frame["engage_date"]).dt.days
    frame["won_value"] = np.where(frame["deal_stage"].eq("Won"), frame["close_value"], 0)
    summary = pd.DataFrame(
        [
            {"dimension": "stage", "level": key, "rows": value}
            for key, value in frame["deal_stage"].value_counts().items()
        ]
        + [
            {"dimension": "sector", "level": str(key), "rows": value}
            for key, value in frame["sector"].fillna("__UNKNOWN__").value_counts().items()
        ]
        + [
            {"dimension": "product", "level": str(key), "rows": value}
            for key, value in frame["product"].value_counts().items()
        ]
    ).assign(protocol_id=PROTOCOL)
    write_table("maven_eda_summary", summary)

    figure_dir = REPO / "artifacts/figures/tracka"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plots = {
        "stage_counts": frame["deal_stage"].value_counts(),
        "won_value_log10": np.log10(frame.loc[frame["won_value"] > 0, "won_value"]),
        "terminal_days": frame.loc[frame["terminal_days"].notna(), "terminal_days"],
        "engagement_month": frame.set_index("engage_date").resample("ME").size(),
    }
    paths = []
    for name, values in plots.items():
        fig, axis = plt.subplots(figsize=(8, 4.5))
        if name in {"stage_counts", "engagement_month"}:
            values.plot(kind="bar" if name == "stage_counts" else "line", ax=axis)
        else:
            axis.hist(values, bins=30)
        axis.set_title(name.replace("_", " ").title())
        axis.grid(alpha=0.2)
        fig.tight_layout()
        path = figure_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path.relative_to(REPO)))
    return {"protocol_id": PROTOCOL, "summary_rows": len(summary), "figures": paths}
