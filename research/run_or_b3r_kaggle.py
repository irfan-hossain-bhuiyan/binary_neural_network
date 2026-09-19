"""B3R: exact B3 replication with verified checkpoint export."""

from __future__ import annotations

import json
import platform
import random
import sys
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    import numpy as np
except ImportError:  # pragma: no cover - Kaggle has NumPy, local import remains usable.
    np = None

from research.run_or_b3 import OPS, run


def main() -> None:
    try:
        seed = 0
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if np is not None:
            np.random.seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = run(epochs=3000, seeds=(0, 1, 2), device=device, export_dir="artifacts/checkpoints")
        payload = {
            "experiment": "B3R-checkpoint-export-replication",
            "parent_experiment": "B3-exact-4bit-XOR",
            "operators": OPS,
            "seeds": [0, 1, 2],
            "epochs": 3000,
            "device": device,
            "architecture": {"input": 8, "width": 64, "blocks": 2, "output": 4},
            "training": {"optimizer": "Adam", "lr": 0.01, "loss": "MSE", "batch_size": 256, "regularizer": None, "noise": None, "scheduler": None, "temperature": None},
            "runtime": {
                "python": platform.python_version(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "python_seed": seed, "numpy_seed": seed, "torch_cpu_seed": seed,
                "torch_cuda_seed": seed if torch.cuda.is_available() else None,
            },
            "results": results,
        }
        Path("b3r_kaggle_metrics.json").write_text(json.dumps(payload, indent=2))
        Path("/kaggle/working/result.json").write_text(json.dumps(payload, indent=2))
    except Exception:
        Path("b3r_error.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
