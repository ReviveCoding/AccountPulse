"""Bounded three-seed Temporal Transformer lifecycle challenger."""

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
from torch.utils.data import DataLoader, TensorDataset

from accountpulse.contracts import capacity_metrics
from accountpulse.evidence import EVIDENCE, write_table
from accountpulse.gpu import GPUSlot, require_cuda
from accountpulse.paths import REPO

SEEDS = [20260922, 20260923, 20260924]
HISTORY = 20
EVENT_DIM = 5


class TemporalTransformer(nn.Module):
    def __init__(
        self, d_model: int = 128, heads: int = 4, layers: int = 2, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(EVENT_DIM, d_model)
        self.position = nn.Parameter(torch.zeros(1, HISTORY, d_model))
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, events: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.projection(events) + self.position, src_key_padding_mask=~mask)
        denominator = mask.sum(1, keepdim=True).clamp_min(1)
        pooled = (encoded * mask.unsqueeze(-1)).sum(1) / denominator
        return self.head(pooled).squeeze(-1)


def make_sequences(baskets: pd.DataFrame, snapshots: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build histories using only events strictly before each snapshot cutoff."""
    baskets = baskets.sort_values(["customer_id", "invoice_date"])
    histories: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for customer, group in baskets.groupby("customer_id", sort=False):
        timestamps = group["invoice_date"].astype("int64").to_numpy()
        gaps = group["invoice_date"].diff().dt.total_seconds().div(86400).fillna(0).clip(0, 365)
        events = np.column_stack(
            [
                np.log1p(group["positive_spend"].clip(lower=0).to_numpy()),
                np.log1p(group["item_count"].clip(lower=0).to_numpy()),
                np.log1p(group["distinct_products"].clip(lower=0).to_numpy()),
                gaps.to_numpy() / 365.0,
                group["cancellation_signal"].astype(float).to_numpy(),
            ]
        ).astype(np.float32)
        histories[str(customer)] = (timestamps, events)
    sequences = np.zeros((len(snapshots), HISTORY, EVENT_DIM), dtype=np.float32)
    masks = np.zeros((len(snapshots), HISTORY), dtype=bool)
    for row_index, row in enumerate(snapshots[["customer_id", "cutoff"]].itertuples(index=False)):
        timestamps, events = histories[str(row.customer_id)]
        boundary = np.searchsorted(timestamps, pd.Timestamp(row.cutoff).value, side="left")
        chosen = events[max(0, boundary - HISTORY) : boundary]
        sequences[row_index, -len(chosen) :] = chosen
        masks[row_index, -len(chosen) :] = True
    return sequences, masks


@dataclass
class SeedResult:
    seed: int
    best_epoch: int
    validation_auc: float
    test_auc: float
    test_brier: float
    value_capture_10: float
    win_capture_10: float
    wall_seconds: float
    peak_vram_bytes: int
    predictions: np.ndarray


def train_seed(
    sequences: np.ndarray,
    masks: np.ndarray,
    snapshots: pd.DataFrame,
    seed: int,
) -> SeedResult:
    device = require_cuda()
    torch.manual_seed(seed)
    np.random.seed(seed)
    fit = snapshots["split"].eq("FIT").to_numpy()
    validation = snapshots["split"].eq("VALIDATION").to_numpy()
    test = snapshots["split"].eq("TEST").to_numpy()
    labels = snapshots["purchase_next_90d"].to_numpy(dtype=np.float32)

    def loader(selection: np.ndarray, shuffle: bool) -> DataLoader[tuple[torch.Tensor, ...]]:
        dataset = TensorDataset(
            torch.from_numpy(sequences[selection]),
            torch.from_numpy(masks[selection]),
            torch.from_numpy(labels[selection]),
        )
        return DataLoader(
            dataset,
            batch_size=512,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
        )

    train_loader = loader(fit, True)
    validation_loader = loader(validation, False)
    test_loader = loader(test, False)
    model = TemporalTransformer().to(device)
    positives = labels[fit].sum()
    positive_weight = torch.tensor([(fit.sum() - positives) / positives], device=device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    best_state: dict[str, torch.Tensor] | None = None
    best_auc = -np.inf
    best_epoch = 0
    patience = 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    def predict(data_loader: DataLoader[tuple[torch.Tensor, ...]]) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        outputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        with torch.no_grad():
            for event, mask, target in data_loader:
                event = event.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    probability = model(event, mask).sigmoid()
                outputs.append(probability.float().cpu().numpy())
                targets.append(target.numpy())
        return np.concatenate(outputs), np.concatenate(targets)

    for epoch in range(1, 13):
        model.train()
        for event, mask, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            event = event.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_function(model(event, mask), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        validation_probability, validation_target = predict(validation_loader)
        auc = roc_auc_score(validation_target, validation_probability)
        if auc > best_auc + 1e-4:
            best_auc = auc
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                break
    if best_state is None:
        raise RuntimeError("Temporal model failed to produce a checkpoint")
    model.load_state_dict(best_state)
    test_probability, test_target = predict(test_loader)
    test_frame = snapshots.loc[test]
    capacity = capacity_metrics(
        test_probability,
        test_frame["future_90d_spend"].to_numpy(dtype=float),
        test_target,
        0.10,
    )
    model_dir = REPO / "artifacts/models/uci"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": best_state, "seed": seed, "history": HISTORY, "event_dim": EVENT_DIM},
        model_dir / f"temporal_transformer_seed_{seed}.pt",
    )
    return SeedResult(
        seed=seed,
        best_epoch=best_epoch,
        validation_auc=float(best_auc),
        test_auc=float(roc_auc_score(test_target, test_probability)),
        test_brier=float(brier_score_loss(test_target, test_probability)),
        value_capture_10=capacity.value_capture,
        win_capture_10=capacity.win_capture,
        wall_seconds=time.perf_counter() - started,
        peak_vram_bytes=int(torch.cuda.max_memory_allocated()),
        predictions=test_probability,
    )


def run() -> dict[str, Any]:
    baskets = pd.read_parquet(EVIDENCE / "uci_baskets.parquet")
    snapshots = pd.read_parquet(EVIDENCE / "uci_lifecycle_snapshots.parquet")
    sequences, masks = make_sequences(baskets, snapshots)
    with GPUSlot(timeout=10):
        results = [train_seed(sequences, masks, snapshots, seed) for seed in SEEDS]
    test = snapshots["split"].eq("TEST").to_numpy()
    ensemble = np.mean([result.predictions for result in results], axis=0)
    test_frame = snapshots.loc[test].reset_index(drop=True)
    capacity = capacity_metrics(
        ensemble,
        test_frame["future_90d_spend"].to_numpy(dtype=float),
        test_frame["purchase_next_90d"].to_numpy(),
        0.10,
    )
    prediction_path = EVIDENCE / "uci_predictions.parquet"
    predictions = pd.read_parquet(prediction_path)
    predictions["temporal_transformer_score"] = ensemble
    write_table("uci_predictions", predictions)
    rows = [
        {
            "track": "C",
            "model": f"G3_temporal_transformer_seed_{result.seed}",
            "split": "TEST",
            "cohort": "ALL",
            "roc_auc": result.test_auc,
            "brier": result.test_brier,
            "log_loss": None,
            "value_capture_10": result.value_capture_10,
            "win_capture_10": result.win_capture_10,
            "selected_count": int(np.ceil(len(test_frame) * 0.10)),
        }
        for result in results
    ]
    rows.append(
        {
            "track": "C",
            "model": "G3_temporal_transformer_ensemble",
            "split": "TEST",
            "cohort": "ALL",
            "roc_auc": roc_auc_score(test_frame["purchase_next_90d"], ensemble),
            "brier": brier_score_loss(test_frame["purchase_next_90d"], ensemble),
            "log_loss": None,
            "value_capture_10": capacity.value_capture,
            "win_capture_10": capacity.win_capture,
            "selected_count": capacity.selected_count,
        }
    )
    metrics_path = EVIDENCE / "ranking_metrics.parquet"
    metrics = pd.read_parquet(metrics_path)
    metrics = metrics.loc[~metrics["model"].astype(str).str.startswith("G3_temporal")]
    write_table("ranking_metrics", pd.concat([metrics, pd.DataFrame(rows)], ignore_index=True))
    system_rows = pd.DataFrame(
        [
            {
                "stage": "E23",
                "model": "G3_temporal_transformer",
                "seed": result.seed,
                "gpu": torch.cuda.get_device_name(0),
                "precision": "BF16_AMP",
                "wall_seconds": result.wall_seconds,
                "peak_vram_bytes": result.peak_vram_bytes,
                "batch_size": 512,
                "sample_count": int((snapshots["split"] == "FIT").sum()),
                "steps": None,
                "single_gpu": True,
            }
            for result in results
        ]
    )
    write_table("systems_metrics", system_rows)
    mlflow.set_tracking_uri(f"sqlite:///{REPO / 'artifacts/evidence/mlflow.db'}")
    mlflow.set_experiment("AccountPulse-E23-Temporal")
    with mlflow.start_run(run_name="temporal_transformer_three_seed"):
        mlflow.log_params(
            {"layers": 2, "d_model": 128, "heads": 4, "dropout": 0.1, "history": HISTORY}
        )
        for result in results:
            mlflow.log_metric(f"seed_{result.seed}.test_auc", result.test_auc)
            mlflow.log_metric(f"seed_{result.seed}.value_capture_10", result.value_capture_10)
    report = {
        "status": "COMPLETE",
        "configuration": {
            "layers": 2,
            "d_model": 128,
            "heads": 4,
            "dropout": 0.1,
            "history": HISTORY,
        },
        "bounded_search_note": (
            "One resource-qualified finalist from the preregistered space; "
            "three independent seeds."
        ),
        "seeds": [
            {key: value for key, value in result.__dict__.items() if key != "predictions"}
            for result in results
        ],
        "ensemble": rows[-1],
        "claim_boundary": "PUBLIC_REAL_WHOLESALE_HEAVY_RETAIL_NOT_CRM",
    }
    (EVIDENCE / "temporal_result.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
