"""Kaggle entry point for the committed B3 four-bit XOR experiment."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from research.run_or_b3 import OPS, run


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = run(epochs=3000, seeds=(0, 1, 2), device=device)
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


if __name__ == "__main__":
    main()
