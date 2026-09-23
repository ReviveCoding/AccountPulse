"""Fail-closed CUDA qualification and one-slot GPU scheduling."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
import xgboost as xgb
from filelock import FileLock, Timeout

from accountpulse.paths import GPU_LOCK_PATH, ensure_runtime_dirs


class GPURequiredError(RuntimeError):
    """Raised when a required CUDA path is not available."""


def require_cuda() -> torch.device:
    """Return CUDA device zero or fail; never silently select CPU."""
    if os.environ.get("ACCOUNT_PULSE_REQUIRE_CUDA", "1") != "1":
        raise GPURequiredError("ACCOUNT_PULSE_REQUIRE_CUDA must remain 1 for GPU stages")
    if not torch.cuda.is_available():
        raise GPURequiredError("CUDA is required but torch.cuda.is_available() is false")
    if torch.cuda.device_count() != 1:
        raise GPURequiredError(
            f"Expected exactly one visible GPU, found {torch.cuda.device_count()}"
        )
    return torch.device("cuda:0")


class GPUSlot(AbstractContextManager["GPUSlot"]):
    """Cross-process lock enforcing one heavy GPU process at a time."""

    def __init__(self, timeout: float = 0.0) -> None:
        ensure_runtime_dirs()
        self._lock = FileLock(GPU_LOCK_PATH, timeout=timeout)

    def __enter__(self) -> Self:
        try:
            self._lock.acquire()
        except Timeout as exc:
            raise GPURequiredError("The single AccountPulse GPU slot is occupied") from exc
        require_cuda()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._lock.release()


def qualify(seed: int = 20260922) -> dict[str, Any]:
    """Run actual Torch FP16 matmul and XGBoost CUDA training."""
    with GPUSlot(timeout=5):
        device = require_cuda()
        torch.manual_seed(seed)
        a = torch.randn((2048, 2048), device=device, dtype=torch.float16)
        b = torch.randn((2048, 2048), device=device, dtype=torch.float16)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        c = a @ b
        torch.cuda.synchronize()
        torch_seconds = time.perf_counter() - started
        finite = bool(torch.isfinite(c).all().item())
        peak_vram = int(torch.cuda.max_memory_allocated())
        del a, b, c
        torch.cuda.empty_cache()

        rng = np.random.default_rng(seed)
        features = rng.normal(size=(20_000, 32)).astype(np.float32)
        target = rng.integers(0, 2, size=20_000, dtype=np.int32)
        model = xgb.XGBClassifier(
            n_estimators=24,
            max_depth=5,
            learning_rate=0.15,
            tree_method="hist",
            device="cuda",
            random_state=seed,
            n_jobs=1,
        )
        started = time.perf_counter()
        model.fit(features, target)
        xgb_seconds = time.perf_counter() - started
        probabilities = model.predict_proba(features[:256])[:, 1]

        return {
            "schema_version": 1,
            "timestamp_utc": __import__("datetime")
            .datetime.now(__import__("datetime").timezone.utc)
            .isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda_available": True,
            "torch_cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "precision": "fp16",
            "torch_fp16_matmul_seconds": torch_seconds,
            "torch_result_finite": finite,
            "peak_vram_bytes": peak_vram,
            "xgboost": xgb.__version__,
            "xgboost_device": "cuda",
            "xgboost_tree_method": "hist",
            "xgboost_cuda_fit_seconds": xgb_seconds,
            "xgboost_predictions_finite": bool(np.isfinite(probabilities).all()),
            "sample_count": len(features),
            "single_gpu_claim_only": True,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = qualify()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
