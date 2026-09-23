"""AccountPulse batch scoring command line interface."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from accountpulse.evidence import EVIDENCE
from accountpulse.pipeline import main as pipeline_main


def score_uci_replay(as_of: str, capacity: float, output: Path | None) -> pd.DataFrame:
    if not 0 < capacity <= 1:
        raise ValueError("capacity must be in (0, 1]")
    predictions = pd.read_parquet(EVIDENCE / "uci_predictions.parquet")
    uncertainty = pd.read_parquet(EVIDENCE / "ranking_uncertainty.parquet")
    timestamp = pd.Timestamp(as_of, tz="UTC")
    if not bool((pd.to_datetime(predictions["cutoff"], utc=True) == timestamp).all()):
        raise ValueError("Only the historically evaluated UCI replay cutoff is available")
    scored = predictions.merge(uncertainty, on="customer_id", validate="1:1")
    scored = scored.sort_values("g6_fixed_late_fusion", ascending=False, kind="stable").reset_index(
        drop=True
    )
    scored["rank"] = np.arange(1, len(scored) + 1)
    scored["selected"] = scored["rank"] <= max(1, int(np.ceil(capacity * len(scored))))
    result = pd.DataFrame(
        {
            "opportunity_id": pd.Series([pd.NA] * len(scored), dtype="string"),
            "account_id": scored["customer_id"].astype("string"),
            "win_probability": scored["p2_adaptive_fusion"],
            "expected_terminal_days": pd.Series([pd.NA] * len(scored), dtype="Float64"),
            "expected_value": pd.Series([pd.NA] * len(scored), dtype="Float64"),
            "accountpulse_ev_score": scored["g6_fixed_late_fusion"],
            "rank": scored["rank"],
            "p_top10": scored["p_top10"],
            "review_flag": scored["review_flag"],
            "selected": scored["selected"],
            "model_version": "uci-replay-fixed-fusion-v1",
            "evidence_class": "HISTORICAL_REPLAY_PUBLIC_RETAIL_NOT_CRM",
        }
    )
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(prog="accountpulse")
    parser.add_argument("--version", action="version", version=version("accountpulse"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--as-of", required=True)
    score.add_argument("--capacity", type=float, default=0.10)
    score.add_argument("--track", choices=["uci-replay"], default="uci-replay")
    score.add_argument("--output", type=Path)
    subparsers.add_parser("pipeline")
    args, remainder = parser.parse_known_args()
    if args.command == "score":
        result = score_uci_replay(args.as_of, args.capacity, args.output)
        if args.output is None:
            print(result.to_csv(index=False))
    else:
        import sys

        sys.argv = ["python -m accountpulse.pipeline", *remainder]
        pipeline_main()


if __name__ == "__main__":
    main()
