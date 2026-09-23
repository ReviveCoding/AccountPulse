"""Cached product-description embeddings and a GPU text-only lifecycle MLP."""

from __future__ import annotations

import json
import time
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import brier_score_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from accountpulse.contracts import capacity_metrics
from accountpulse.evidence import EVIDENCE, write_table
from accountpulse.gpu import GPUSlot, require_cuda
from accountpulse.paths import REPO

SEED = 20260922
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def item_embeddings(lines: pd.DataFrame) -> tuple[dict[str, int], np.ndarray]:
    cache = REPO / "data/processed/uci_item_embeddings.npz"
    items_path = REPO / "data/processed/uci_item_embedding_index.parquet"
    if cache.exists() and items_path.exists():
        index = pd.read_parquet(items_path)
        return dict(zip(index["stock_code"], index["row"], strict=True)), np.load(cache)[
            "embeddings"
        ]
    items = (
        lines.dropna(subset=["description"])
        .sort_values("invoice_date")
        .drop_duplicates("stock_code", keep="first")[["stock_code", "description"]]
        .reset_index(drop=True)
    )
    encoder = SentenceTransformer(MODEL_NAME, device="cuda")
    embeddings = encoder.encode(
        items["description"].astype(str).tolist(),
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, embeddings=embeddings)
    items.assign(row=np.arange(len(items))).to_parquet(items_path, index=False)
    return dict(zip(items["stock_code"], np.arange(len(items)), strict=True)), embeddings


def snapshot_embeddings(
    lines: pd.DataFrame,
    snapshots: pd.DataFrame,
    item_index: dict[str, int],
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cache = REPO / "data/processed/uci_snapshot_text_embeddings.npz"
    if cache.exists():
        stored = np.load(cache)
        return stored["embeddings"], stored["missing"]
    valid = lines[lines["stock_code"].isin(item_index)].copy()
    valid["embedding_row"] = valid["stock_code"].map(item_index)
    customer_events: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for customer, group in valid.sort_values("invoice_date").groupby("customer_id", sort=False):
        customer_events[str(customer)] = (
            group["invoice_date"].astype("int64").to_numpy(),
            group["embedding_row"].to_numpy(dtype=int),
        )
    output = np.zeros((len(snapshots), embeddings.shape[1]), dtype=np.float32)
    missing = np.ones(len(snapshots), dtype=np.float32)
    for position, row in enumerate(snapshots[["customer_id", "cutoff"]].itertuples(index=False)):
        events = customer_events.get(str(row.customer_id))
        if events is None:
            continue
        timestamps, indices = events
        boundary = np.searchsorted(timestamps, pd.Timestamp(row.cutoff).value, side="left")
        if boundary:
            recent = indices[max(0, boundary - 200) : boundary]
            output[position] = embeddings[recent].mean(axis=0)
            missing[position] = 0.0
    np.savez_compressed(cache, embeddings=output, missing=missing)
    return output, missing


class TextMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width + 1, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def run() -> dict[str, Any]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    lines = pd.read_parquet(REPO / "data/processed/uci_lines.parquet")
    snapshots = pd.read_parquet(EVIDENCE / "uci_lifecycle_snapshots.parquet")
    with GPUSlot(timeout=10):
        device = require_cuda()
        index, items = item_embeddings(lines)
        vectors, missing = snapshot_embeddings(lines, snapshots, index, items)
        features = np.column_stack([vectors, missing]).astype(np.float32)
        labels = snapshots["purchase_next_90d"].to_numpy(dtype=np.float32)
        fit = snapshots["split"].eq("FIT").to_numpy()
        validation = snapshots["split"].eq("VALIDATION").to_numpy()
        test = snapshots["split"].eq("TEST").to_numpy()

        def loader(mask: np.ndarray, shuffle: bool) -> DataLoader[tuple[torch.Tensor, ...]]:
            return DataLoader(
                TensorDataset(torch.from_numpy(features[mask]), torch.from_numpy(labels[mask])),
                batch_size=1024,
                shuffle=shuffle,
                pin_memory=True,
            )

        model = TextMLP(items.shape[1]).to(device)
        positives = labels[fit].sum()
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([(fit.sum() - positives) / positives], device=device)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        best_auc = -np.inf
        best_state: dict[str, torch.Tensor] | None = None
        patience = 0
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

        def predict(data: DataLoader[tuple[torch.Tensor, ...]]) -> tuple[np.ndarray, np.ndarray]:
            model.eval()
            probabilities, targets = [], []
            with torch.no_grad():
                for batch, target in data:
                    probability = model(batch.to(device, non_blocking=True)).sigmoid()
                    probabilities.append(probability.cpu().numpy())
                    targets.append(target.numpy())
            return np.concatenate(probabilities), np.concatenate(targets)

        for _epoch in range(20):
            model.train()
            for batch, target in loader(fit, True):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch.to(device)), target.to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            probability, target = predict(loader(validation, False))
            auc = roc_auc_score(target, probability)
            if auc > best_auc + 1e-4:
                best_auc = auc
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= 3:
                    break
        if best_state is None:
            raise RuntimeError("Text MLP produced no valid checkpoint")
        model.load_state_dict(best_state)
        probability, target = predict(loader(test, False))
        test_frame = snapshots.loc[test].reset_index(drop=True)
        capacity = capacity_metrics(
            probability,
            test_frame["future_90d_spend"].to_numpy(float),
            target,
            0.10,
        )
        peak = int(torch.cuda.max_memory_allocated())
        wall = time.perf_counter() - started
    model_dir = REPO / "artifacts/models/uci"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "model_name": MODEL_NAME}, model_dir / "text_mlp.pt")
    predictions = pd.read_parquet(EVIDENCE / "uci_predictions.parquet")
    predictions["text_score"] = probability
    write_table("uci_predictions", predictions)
    row = {
        "track": "C",
        "model": "G5_text_embedding_mlp",
        "split": "TEST",
        "cohort": "ALL",
        "roc_auc": roc_auc_score(target, probability),
        "brier": brier_score_loss(target, probability),
        "log_loss": None,
        "value_capture_10": capacity.value_capture,
        "win_capture_10": capacity.win_capture,
        "selected_count": capacity.selected_count,
    }
    metrics = pd.read_parquet(EVIDENCE / "ranking_metrics.parquet")
    metrics = metrics.loc[metrics["model"] != row["model"]]
    write_table("ranking_metrics", pd.concat([metrics, pd.DataFrame([row])], ignore_index=True))
    systems_path = EVIDENCE / "systems_metrics.parquet"
    systems = pd.read_parquet(systems_path) if systems_path.exists() else pd.DataFrame()
    systems = (
        systems.loc[systems.get("stage", pd.Series(dtype=str)) != "E25"]
        if not systems.empty
        else systems
    )
    system = {
        "stage": "E25",
        "model": "G5_text_embedding_mlp",
        "seed": SEED,
        "gpu": torch.cuda.get_device_name(0),
        "precision": "FP32",
        "wall_seconds": wall,
        "peak_vram_bytes": peak,
        "batch_size": 1024,
        "sample_count": int(fit.sum()),
        "steps": None,
        "single_gpu": True,
    }
    write_table("systems_metrics", pd.concat([systems, pd.DataFrame([system])], ignore_index=True))
    mlflow.set_tracking_uri(f"sqlite:///{REPO / 'artifacts/evidence/mlflow.db'}")
    mlflow.set_experiment("AccountPulse-E25-Text")
    with mlflow.start_run(run_name="uci_cached_text_mlp"):
        mlflow.log_params({"encoder": MODEL_NAME, "seed": SEED, "cache_immutable": True})
        mlflow.log_metrics(
            {key: float(value) for key, value in row.items() if isinstance(value, (int, float))}
        )
    report = {
        "status": "COMPLETE",
        "item_count": len(index),
        "embedding_dimension": int(items.shape[1]),
        "missing_snapshot_fraction": float(missing.mean()),
        "validation_auc": float(best_auc),
        "metrics": row,
        "systems": system,
        "claim_boundary": "PUBLIC_REAL_WHOLESALE_HEAVY_RETAIL_NOT_CRM",
    }
    (EVIDENCE / "text_result.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
