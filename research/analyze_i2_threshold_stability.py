"""Post-process I2 checkpoints for functional threshold robustness.

This is deliberately a no-training diagnostic.  I2 used 0.5 as the
canonical Boolean threshold; this script sweeps nearby thresholds on the
same complete truth table and records contiguous exact-function intervals.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.boolean_tasks import build_task  # noqa: E402
from research.run_initialization_i2 import model_for  # noqa: E402


DEFAULT_THRESHOLDS = [round(0.20 + 0.025 * i, 3) for i in range(25)]
DEFAULT_KINDS = ("best_boolean_exact", "best_continuous_mse", "final")


def contiguous_intervals(exact_thresholds: list[float]) -> list[list[float]]:
    """Return maximal contiguous runs from an ordered threshold grid."""
    exact = set(exact_thresholds)
    grid = DEFAULT_THRESHOLDS
    intervals: list[list[float]] = []
    start: float | None = None
    previous: float | None = None
    step = 0.025
    for threshold in grid:
        if threshold in exact:
            if start is None or previous is None or abs(threshold - previous - step) > 1e-9:
                if start is not None and previous is not None:
                    intervals.append([start, previous])
                start = threshold
            previous = threshold
        elif start is not None and previous is not None:
            intervals.append([start, previous])
            start = previous = None
    if start is not None and previous is not None:
        intervals.append([start, previous])
    return intervals


def threshold_metrics(model, x: torch.Tensor, y: torch.Tensor, threshold: float) -> dict:
    with torch.no_grad():
        discrete = model.to_discrete(threshold)
        output = discrete(x.bool()).float()
        correct = (output == y).all(dim=-1)
        return {
            "bit_accuracy": float((output == y).float().mean()),
            "exact_accuracy": float(correct.float().mean()),
            "mse": float((output - y).square().mean()),
        }


def report_checkpoint(condition: str, seed: int, kind: str, path: Path,
                      x: torch.Tensor, y: torch.Tensor,
                      thresholds: list[float]) -> dict:
    model = model_for(condition, seed, torch.device("cpu"))
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    values = {
        str(threshold): threshold_metrics(model, x, y, threshold)
        for threshold in thresholds
    }
    exact = [threshold for threshold in thresholds
             if values[str(threshold)]["exact_accuracy"] == 1.0]
    # The canonical grid is used for interval calculation.  For custom
    # grids, the exact list remains available even if an interval is empty.
    intervals = contiguous_intervals(exact) if thresholds == DEFAULT_THRESHOLDS else []
    largest = max(intervals, key=lambda pair: pair[1] - pair[0], default=None)
    return {
        "checkpoint": str(path),
        "kind": kind,
        "thresholds": values,
        "exact_thresholds": exact,
        "all_contiguous_intervals": intervals,
        "largest_contiguous_interval": largest,
        "functional_threshold_margin": (largest[1] - largest[0]) if largest else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(ROOT / "research/operator_results/initialization_i2_results.json"))
    parser.add_argument("--checkpoints", default=str(ROOT / "research/operator_results/initialization_i2_checkpoints"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result_path = Path(args.results)
    checkpoint_root = Path(args.checkpoints)
    payload = json.loads(result_path.read_text())
    task = build_task("bitwise_xor_truth_table", {"bits": 4})
    x, y = task["X"].float(), task["Y"].float()
    thresholds = DEFAULT_THRESHOLDS

    updated = copy.deepcopy(payload)
    for row in updated["results"]:
        condition, seed = row["condition"], int(row["seed"])
        reports = {}
        for kind in DEFAULT_KINDS:
            path = checkpoint_root / f"I2_{condition}_seed{seed}_{kind}.pt"
            if path.exists():
                reports[kind] = report_checkpoint(condition, seed, kind, path, x, y, thresholds)
        row["threshold_stability"] = reports

    updated["threshold_stability_grid"] = thresholds
    output = Path(args.output) if args.output else result_path
    output.write_text(json.dumps(updated, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
