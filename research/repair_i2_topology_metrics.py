"""Repair I2 final-topology bookkeeping from saved final checkpoints.

The original I2 runner calculated final mask fields after loading the best
continuous checkpoint.  This script recomputes hashes from the actual final
checkpoint without inventing Hamming distances when the initial raw tensors
are unavailable.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import run_initialization_i2 as base  # noqa: E402


def main() -> None:
    result_path = ROOT / "research/operator_results/initialization_i2_results.json"
    checkpoint_root = ROOT / "research/operator_results/initialization_i2_checkpoints"
    payload = json.loads(result_path.read_text())
    updated = copy.deepcopy(payload)
    for row in updated["results"]:
        condition, seed = row["condition"], int(row["seed"])
        path = checkpoint_root / f"I2_{condition}_seed{seed}_final.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        model = base.model_for(condition, seed, torch.device("cpu"))
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        repaired = {
            "source_checkpoint": str(path),
            "edge_mask_sha256": base.mask_hash(model),
            "bias_mask_sha256": base.mask_hash(model, True),
            "edge_mask_hamming": None,
            "bias_mask_hamming": None,
            "hamming_status": "not computed: initial raw state was not archived in I2",
            "layer_stats": base.layer_stats(model),
        }
        row["topology_repair"] = repaired
    updated["topology_repair_note"] = (
        "The original final_mask_sha256 and Hamming fields were calculated "
        "after loading best_continuous. Use topology_repair for true final "
        "checkpoint topology; Hamming distances remain null because I2 did "
        "not archive initial raw tensors."
    )
    result_path.write_text(json.dumps(updated, indent=2) + "\n")
    print(result_path)


if __name__ == "__main__":
    main()
