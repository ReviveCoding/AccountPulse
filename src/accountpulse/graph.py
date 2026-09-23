"""PIT-correct customer-product GraphSAGE lifecycle challenger."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score
from torch import nn
from torch_geometric.nn import SAGEConv

from accountpulse.contracts import capacity_metrics
from accountpulse.evidence import EVIDENCE, write_table
from accountpulse.gpu import GPUSlot, require_cuda
from accountpulse.paths import REPO

SEED = 20260922


@dataclass
class GraphFrame:
    features: torch.Tensor
    edge_index: torch.Tensor
    customer_nodes: torch.Tensor
    labels: torch.Tensor
    snapshot_rows: pd.DataFrame


class GraphSAGE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = SAGEConv(6, 64)
        self.second = SAGEConv(64, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, features: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        hidden = self.first(features, edges).relu()
        hidden = nn.functional.dropout(hidden, p=0.2, training=self.training)
        hidden = self.second(hidden, edges).relu()
        return self.output(hidden).squeeze(-1)


def graph_at_cutoff(
    lines: pd.DataFrame, snapshots: pd.DataFrame, cutoff: pd.Timestamp
) -> GraphFrame:
    snapshot = snapshots[snapshots["cutoff"] == cutoff].copy().reset_index(drop=True)
    history = lines[
        (lines["invoice_date"] < cutoff) & (~lines["cancelled"]) & (lines["quantity"] > 0)
    ].copy()
    history = history[history["customer_id"].isin(snapshot["customer_id"])]
    customer_ids = snapshot["customer_id"].astype(str).tolist()
    item_ids = sorted(history["stock_code"].astype(str).unique())
    customer_index = {identifier: position for position, identifier in enumerate(customer_ids)}
    item_index = {
        identifier: len(customer_ids) + position for position, identifier in enumerate(item_ids)
    }
    edges = history[["customer_id", "stock_code"]].drop_duplicates()
    sources = edges["customer_id"].astype(str).map(customer_index).to_numpy(dtype=np.int64)
    targets = edges["stock_code"].astype(str).map(item_index).to_numpy(dtype=np.int64)
    edge_array = np.column_stack(
        [np.concatenate([sources, targets]), np.concatenate([targets, sources])]
    ).T
    node_count = len(customer_ids) + len(item_ids)
    features = np.zeros((node_count, 6), dtype=np.float32)
    features[: len(customer_ids), 0] = 1.0
    customer_stats = history.groupby("customer_id").agg(
        count=("invoice", "nunique"),
        quantity=("quantity", "sum"),
        spend=("line_value", "sum"),
        breadth=("stock_code", "nunique"),
    )
    for identifier, position in customer_index.items():
        if identifier in customer_stats.index:
            values = customer_stats.loc[identifier]
            features[position, 2:] = np.log1p(
                [values["count"], values["quantity"], max(values["spend"], 0), values["breadth"]]
            )
    features[len(customer_ids) :, 1] = 1.0
    item_stats = history.groupby("stock_code").agg(
        customers=("customer_id", "nunique"),
        quantity=("quantity", "sum"),
        spend=("line_value", "sum"),
        invoices=("invoice", "nunique"),
    )
    for identifier, position in item_index.items():
        values = item_stats.loc[identifier]
        features[position, 2:] = np.log1p(
            [values["customers"], values["quantity"], max(values["spend"], 0), values["invoices"]]
        )
    features[:, 2:] = (features[:, 2:] - features[:, 2:].mean(0)) / (features[:, 2:].std(0) + 1e-6)
    return GraphFrame(
        features=torch.from_numpy(features),
        edge_index=torch.from_numpy(edge_array),
        customer_nodes=torch.arange(len(customer_ids)),
        labels=torch.from_numpy(snapshot["purchase_next_90d"].to_numpy(np.float32)),
        snapshot_rows=snapshot,
    )


def run() -> dict[str, Any]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    lines = pd.read_parquet(REPO / "data/processed/uci_lines.parquet")
    snapshots = pd.read_parquet(EVIDENCE / "uci_lifecycle_snapshots.parquet")
    fit_cutoff = snapshots.loc[snapshots["split"] == "FIT", "cutoff"].max()
    validation_cutoff = snapshots.loc[snapshots["split"] == "VALIDATION", "cutoff"].iloc[0]
    test_cutoff = snapshots.loc[snapshots["split"] == "TEST", "cutoff"].iloc[0]
    train_graph = graph_at_cutoff(lines, snapshots, fit_cutoff)
    validation_graph = graph_at_cutoff(lines, snapshots, validation_cutoff)
    test_graph = graph_at_cutoff(lines, snapshots, test_cutoff)
    with GPUSlot(timeout=10):
        device = require_cuda()
        model = GraphSAGE().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
        positive = train_graph.labels.sum().item()
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [(len(train_graph.labels) - positive) / positive], device=device
            )
        )
        best_auc = -np.inf
        best_state: dict[str, torch.Tensor] | None = None
        patience = 0
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

        def logits(frame: GraphFrame) -> torch.Tensor:
            return model(frame.features.to(device), frame.edge_index.to(device))[
                frame.customer_nodes.to(device)
            ]

        for _epoch in range(100):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(logits(train_graph), train_graph.labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                probability = logits(validation_graph).sigmoid().cpu().numpy()
            auc = roc_auc_score(validation_graph.labels.numpy(), probability)
            if auc > best_auc + 1e-4:
                best_auc = auc
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= 10:
                    break
        if best_state is None:
            raise RuntimeError("GraphSAGE produced no valid checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            probability = logits(test_graph).sigmoid().cpu().numpy()
        peak = int(torch.cuda.max_memory_allocated())
        wall = time.perf_counter() - started
    test_target = test_graph.labels.numpy()
    capacity = capacity_metrics(
        probability,
        test_graph.snapshot_rows["future_90d_spend"].to_numpy(float),
        test_target,
        0.10,
    )
    row = {
        "track": "C",
        "model": "G4_graphsage",
        "split": "TEST",
        "cohort": "ALL",
        "roc_auc": roc_auc_score(test_target, probability),
        "brier": brier_score_loss(test_target, probability),
        "log_loss": None,
        "value_capture_10": capacity.value_capture,
        "win_capture_10": capacity.win_capture,
        "selected_count": capacity.selected_count,
    }
    predictions = pd.read_parquet(EVIDENCE / "uci_predictions.parquet")
    graph_scores = pd.DataFrame(
        {"customer_id": test_graph.snapshot_rows["customer_id"], "graphsage_score": probability}
    )
    predictions = predictions.drop(columns=["graphsage_score"], errors="ignore").merge(
        graph_scores, on="customer_id", how="left", validate="1:1"
    )
    write_table("uci_predictions", predictions)
    metrics = pd.read_parquet(EVIDENCE / "ranking_metrics.parquet")
    metrics = metrics.loc[metrics["model"] != row["model"]]
    write_table("ranking_metrics", pd.concat([metrics, pd.DataFrame([row])], ignore_index=True))
    system = {
        "stage": "E24",
        "model": "G4_graphsage",
        "seed": SEED,
        "gpu": torch.cuda.get_device_name(0),
        "precision": "FP32",
        "wall_seconds": wall,
        "peak_vram_bytes": peak,
        "batch_size": len(train_graph.labels),
        "sample_count": len(train_graph.labels),
        "steps": None,
        "single_gpu": True,
    }
    systems = pd.read_parquet(EVIDENCE / "systems_metrics.parquet")
    systems = systems.loc[systems["stage"] != "E24"]
    write_table("systems_metrics", pd.concat([systems, pd.DataFrame([system])], ignore_index=True))
    model_dir = REPO / "artifacts/models/uci"
    torch.save(
        {"state_dict": best_state, "features": 6, "fit_cutoff": str(fit_cutoff)},
        model_dir / "graphsage.pt",
    )
    mlflow.set_tracking_uri(f"sqlite:///{REPO / 'artifacts/evidence/mlflow.db'}")
    mlflow.set_experiment("AccountPulse-E24-GraphSAGE")
    with mlflow.start_run(run_name="uci_pit_graphsage"):
        mlflow.log_params(
            {
                "seed": SEED,
                "fit_cutoff": str(fit_cutoff),
                "validation_cutoff": str(validation_cutoff),
                "test_cutoff": str(test_cutoff),
            }
        )
        mlflow.log_metrics(
            {key: float(value) for key, value in row.items() if isinstance(value, (int, float))}
        )
    report = {
        "status": "COMPLETE",
        "fit_cutoff": str(fit_cutoff),
        "validation_cutoff": str(validation_cutoff),
        "test_cutoff": str(test_cutoff),
        "future_edges_forbidden": True,
        "validation_auc": float(best_auc),
        "metrics": row,
        "systems": system,
        "claim_boundary": "PUBLIC_REAL_WHOLESALE_HEAVY_RETAIL_NOT_CRM",
    }
    (EVIDENCE / "graph_result.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
