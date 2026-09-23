"""Validation-trained adaptive fusion and mandatory modality ablations."""

from __future__ import annotations

import json
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from accountpulse.contracts import capacity_metrics
from accountpulse.evidence import EVIDENCE, write_table
from accountpulse.gpu import GPUSlot, require_cuda
from accountpulse.graph import GraphSAGE, graph_at_cutoff
from accountpulse.paths import REPO
from accountpulse.temporal import TemporalTransformer, make_sequences
from accountpulse.text_model import TextMLP
from accountpulse.uci import FEATURES

SEED = 20260922
MODALITIES = ["tabular", "temporal", "graph", "text"]


class AdaptiveGate(nn.Module):
    def __init__(self, state_width: int, modality_count: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(state_width, 16), nn.GELU(), nn.Linear(16, modality_count)
        )

    def forward(self, scores: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        weights = self.gate(state).softmax(dim=1)
        return (weights * scores).sum(dim=1)


def _temporal_scores(
    sequences: np.ndarray,
    masks: np.ndarray,
    selection: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    event = torch.from_numpy(sequences[selection]).to(device)
    mask = torch.from_numpy(masks[selection]).to(device)
    outputs = []
    for seed in (20260922, 20260923, 20260924):
        checkpoint = torch.load(
            REPO / f"artifacts/models/uci/temporal_transformer_seed_{seed}.pt",
            map_location=device,
            weights_only=True,
        )
        model = TemporalTransformer().to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        batches = []
        with torch.no_grad():
            for start in range(0, len(event), 1024):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    batches.append(
                        model(event[start : start + 1024], mask[start : start + 1024])
                        .sigmoid()
                        .float()
                        .cpu()
                        .numpy()
                    )
        outputs.append(np.concatenate(batches))
    return np.mean(outputs, axis=0)


def _graph_score(
    lines: pd.DataFrame,
    snapshots: pd.DataFrame,
    cutoff: pd.Timestamp,
    device: torch.device,
) -> np.ndarray:
    frame = graph_at_cutoff(lines, snapshots, cutoff)
    checkpoint = torch.load(
        REPO / "artifacts/models/uci/graphsage.pt", map_location=device, weights_only=True
    )
    model = GraphSAGE().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        return (
            model(frame.features.to(device), frame.edge_index.to(device))[
                frame.customer_nodes.to(device)
            ]
            .sigmoid()
            .cpu()
            .numpy()
        )


def _text_scores(
    vectors: np.ndarray, missing: np.ndarray, selection: np.ndarray, device: torch.device
) -> np.ndarray:
    checkpoint = torch.load(
        REPO / "artifacts/models/uci/text_mlp.pt", map_location=device, weights_only=True
    )
    model = TextMLP(vectors.shape[1]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    features = torch.from_numpy(
        np.column_stack([vectors[selection], missing[selection]]).astype(np.float32)
    )
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), 2048):
            outputs.append(model(features[start : start + 2048].to(device)).sigmoid().cpu().numpy())
    return np.concatenate(outputs)


def _state(frame: pd.DataFrame, missing: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.log1p(frame["frequency"].to_numpy(float)),
            np.log1p(frame["history_days"].to_numpy(float)),
            missing,
            frame["customer_cohort"].eq("UNSEEN").to_numpy(float),
            frame["recency_days"].to_numpy(float) / 365.0,
        ]
    ).astype(np.float32)


def _train_gate(
    validation_scores: np.ndarray,
    validation_state: np.ndarray,
    validation_target: np.ndarray,
    test_scores: np.ndarray,
    test_state: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    torch.manual_seed(SEED)
    model = AdaptiveGate(validation_state.shape[1], validation_scores.shape[1]).to(device)
    score = torch.from_numpy(validation_scores.astype(np.float32)).to(device)
    state = torch.from_numpy(validation_state).to(device)
    target = torch.from_numpy(validation_target.astype(np.float32)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        probability = model(score, state).clamp(1e-5, 1 - 1e-5)
        loss = nn.functional.binary_cross_entropy(probability, target)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return (
            model(
                torch.from_numpy(test_scores.astype(np.float32)).to(device),
                torch.from_numpy(test_state).to(device),
            )
            .cpu()
            .numpy()
        )


def _row(name: str, test: pd.DataFrame, score: np.ndarray) -> dict[str, Any]:
    y = test["purchase_next_90d"].to_numpy()
    capacity = capacity_metrics(score, test["future_90d_spend"].to_numpy(float), y, 0.10)
    return {
        "track": "C",
        "model": name,
        "split": "TEST",
        "cohort": "ALL",
        "roc_auc": roc_auc_score(y, score),
        "brier": brier_score_loss(y, score),
        "log_loss": None,
        "value_capture_10": capacity.value_capture,
        "win_capture_10": capacity.win_capture,
        "selected_count": capacity.selected_count,
    }


def run() -> dict[str, Any]:
    np.random.seed(SEED)
    snapshots = pd.read_parquet(EVIDENCE / "uci_lifecycle_snapshots.parquet")
    baskets = pd.read_parquet(EVIDENCE / "uci_baskets.parquet")
    lines = pd.read_parquet(REPO / "data/processed/uci_lines.parquet")
    vectors = np.load(REPO / "data/processed/uci_snapshot_text_embeddings.npz")
    text_vectors, text_missing = vectors["embeddings"], vectors["missing"]
    sequences, masks = make_sequences(baskets, snapshots)
    validation_mask = snapshots["split"].eq("VALIDATION").to_numpy()
    test_mask = snapshots["split"].eq("TEST").to_numpy()
    validation = snapshots.loc[validation_mask].reset_index(drop=True)
    test = snapshots.loc[test_mask].reset_index(drop=True)
    encoder, boosted = joblib.load(REPO / "artifacts/models/uci/xgboost_cuda.joblib")
    validation_tabular = boosted.predict_proba(encoder.transform(validation[FEATURES]))[:, 1]
    test_tabular = boosted.predict_proba(encoder.transform(test[FEATURES]))[:, 1]
    validation_cutoff = validation["cutoff"].iloc[0]
    test_cutoff = test["cutoff"].iloc[0]
    with GPUSlot(timeout=10):
        device = require_cuda()
        validation_temporal = _temporal_scores(sequences, masks, validation_mask, device)
        test_temporal = _temporal_scores(sequences, masks, test_mask, device)
        validation_graph = _graph_score(lines, snapshots, validation_cutoff, device)
        test_graph = _graph_score(lines, snapshots, test_cutoff, device)
        validation_text = _text_scores(text_vectors, text_missing, validation_mask, device)
        test_text = _text_scores(text_vectors, text_missing, test_mask, device)
        validation_scores = np.column_stack(
            [validation_tabular, validation_temporal, validation_graph, validation_text]
        )
        test_scores = np.column_stack([test_tabular, test_temporal, test_graph, test_text])
        validation_state = _state(validation, text_missing[validation_mask])
        test_state = _state(test, text_missing[test_mask])
        scores: dict[str, np.ndarray] = {"G6_fixed_late_fusion": test_scores.mean(axis=1)}
        scores["P2_adaptive_fusion"] = _train_gate(
            validation_scores,
            validation_state,
            validation["purchase_next_90d"].to_numpy(),
            test_scores,
            test_state,
            device,
        )
        for index, modality in enumerate(MODALITIES):
            keep = [position for position in range(len(MODALITIES)) if position != index]
            scores[f"P2_minus_{modality}"] = _train_gate(
                validation_scores[:, keep],
                validation_state,
                validation["purchase_next_90d"].to_numpy(),
                test_scores[:, keep],
                test_state,
                device,
            )
    # Early tabular+text concatenation is fit on FIT only and remains CPU-regularized.
    fit_mask = snapshots["split"].eq("FIT").to_numpy()
    early_features = np.column_stack(
        [
            snapshots[["recency_days", "frequency", "monetary", "product_breadth"]].to_numpy(float),
            text_vectors,
        ]
    )
    early = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)),
        ]
    )
    early.fit(early_features[fit_mask], snapshots.loc[fit_mask, "purchase_next_90d"])
    scores["G6_early_fusion_tabular_text"] = early.predict_proba(early_features[test_mask])[:, 1]
    rows = [_row(name, test, score) for name, score in scores.items()]
    predictions = pd.read_parquet(EVIDENCE / "uci_predictions.parquet")
    for name, score in scores.items():
        predictions[name.lower()] = score
    write_table("uci_predictions", predictions)
    metrics = pd.read_parquet(EVIDENCE / "ranking_metrics.parquet")
    metrics = metrics.loc[~metrics["model"].astype(str).str.startswith(("G6_", "P2_"))]
    write_table("ranking_metrics", pd.concat([metrics, pd.DataFrame(rows)], ignore_index=True))
    joblib.dump(early, REPO / "artifacts/models/uci/early_fusion.joblib")
    mlflow.set_tracking_uri(f"sqlite:///{REPO / 'artifacts/evidence/mlflow.db'}")
    mlflow.set_experiment("AccountPulse-E26-E27-Fusion")
    with mlflow.start_run(run_name="uci_fusion_and_ablations"):
        mlflow.log_params(
            {"gate_evidence": "VALIDATION_ONLY", "modalities": MODALITIES, "seed": SEED}
        )
        for index, row in enumerate(rows):
            mlflow.log_metric(f"row{index}.value_capture_10", row["value_capture_10"])
            mlflow.log_metric(f"row{index}.roc_auc", row["roc_auc"])
    report = {
        "status": "COMPLETE",
        "gate_training_split": "VALIDATION",
        "test_weight_tuning": False,
        "modalities": MODALITIES,
        "metrics": rows,
        "claim_boundary": "PUBLIC_REAL_WHOLESALE_HEAVY_RETAIL_NOT_CRM",
    }
    (EVIDENCE / "fusion_result.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
