"""Kaggle entry point for the committed B3 four-bit XOR experiment."""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path

import torch

# Kaggle executes this file directly, so Python initially puts only the
# ``research`` directory on sys.path.  Add the packaged repository root so
# imports work both locally and in the remote kernel.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from research.run_or_b3 import OPS, run


def main() -> None:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = run(epochs=3000, seeds=(0, 1, 2), device=device)
        # The Kaggle bootstrap exports this directory after the entry point
        # returns.  Keep the exact B3 checkpoints available for no-retraining
        # forensic analysis instead of losing them with the source tree.
        source_checkpoints = Path("research/operator_results/stage_b3_checkpoints")
        export_checkpoints = Path("artifacts/checkpoints")
        export_checkpoints.mkdir(parents=True, exist_ok=True)
        for checkpoint in source_checkpoints.glob("*.pt"):
            shutil.copy2(checkpoint, export_checkpoints / checkpoint.name)
        payload = {
            "experiment": "B3-exact-4bit-XOR",
            "operators": OPS,
            "seeds": [0, 1, 2],
            "epochs": 3000,
            "device": device,
            "architecture": {"input": 8, "width": 64, "blocks": 2, "output": 4},
            "results": results,
        }
        Path("b3_kaggle_metrics.json").write_text(json.dumps(payload, indent=2))
        Path("/kaggle/working/result.json").write_text(json.dumps(payload, indent=2))
    except Exception:
        Path("b3_error.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
