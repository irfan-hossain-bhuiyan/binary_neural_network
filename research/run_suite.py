"""Sequential task x seed benchmark suite runner (one Kaggle GPU job).

Usage:
    python research/run_suite.py --suite baseline --output metrics.json
    python research/run_suite.py --suite baseline --seeds 0,1,2 --output metrics.json

Output:
    {
      "suite_name": ...,
      "git_commit": ...,
      "runs": [ {experiment_id, task_name, ..., seed, status, metrics|error} ],
      "summary": { per-task aggregates + recovery rates }
    }

A failing task is recorded with status "failed" and the suite continues.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_experiment import (  # noqa: E402
    get_git_commit,
    run_single,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "research" / "configs"

# Numeric metrics aggregated across seeds (mean/std/min/max).
AGGREGATE_METRICS = [
    "continuous_accuracy",
    "continuous_bit_accuracy",
    "continuous_exact_accuracy",
    "discrete_bit_accuracy",
    "discrete_accuracy",
    "discrete_exact_accuracy",
    "continuous_discrete_gap",
    "final_loss",
    "runtime_seconds",
    "selected_edge_fraction",
    "first_last_gradient_ratio",
    "mean_binary_entropy",
]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _collect_metric(run_metrics: dict, key: str) -> list[float]:
    """Fetch a (possibly nested) numeric metric from one run's metrics."""
    if key == "selected_edge_fraction":
        v = (run_metrics.get("circuit_stats") or {}).get("selected_edge_fraction")
    elif key == "first_last_gradient_ratio":
        v = (run_metrics.get("gradient_summary") or {}).get("first_last_gradient_ratio")
    elif key == "mean_binary_entropy":
        acts = run_metrics.get("activation_stats") or {}
        vals: list[float] = []
        for s in acts.values():
            if isinstance(s, dict):
                v = s.get("mean_binary_entropy")
                if isinstance(v, (int, float)):
                    vals.append(float(v))
        return vals
    else:
        v = run_metrics.get(key)
    return [float(v)] if isinstance(v, (int, float)) else []


def summarize_task(task_runs: list[dict]) -> dict:
    ok = [r for r in task_runs if r["status"] == "success"]
    summary: dict = {
        "num_runs": len(task_runs),
        "num_success": len(ok),
        "num_failed": len(task_runs) - len(ok),
    }
    for key in AGGREGATE_METRICS:
        vals: list[float] = []
        for r in ok:
            vals.extend(_collect_metric(r["metrics"], key))
        if vals:
            summary[key] = {
                "mean": _mean(vals),
                "std": _std(vals),
                "min": min(vals),
                "max": max(vals),
                "n": len(vals),
            }
    # Exact Boolean recovery rate: fraction of successful seeds whose
    # DISCRETIZED network reproduces the eval set exactly.
    if ok:
        recovered = sum(1 for r in ok if r["metrics"].get("discrete_function_recovery"))
        summary["discrete_exact_recovery_rate"] = recovered / len(ok)
        summary["discrete_exact_recovery_count"] = f"{recovered}/{len(ok)}"
    return summary


def run_suite(suite_cfg: dict, seeds: list[int], git_commit: str | None) -> dict:
    base_training = dict(suite_cfg.get("training", {}))
    base_model = dict(suite_cfg.get("model", {}))
    threshold = float(suite_cfg.get("discretization_threshold", 0.5))
    epochs = suite_cfg.get("epochs")
    batch_size = suite_cfg.get("batch_size")
    if epochs is not None:
        base_training["epochs"] = int(epochs)
    if batch_size is not None:
        base_training["batch_size"] = int(batch_size)

    runs: list[dict] = []
    t0 = time.time()
    for task_entry in suite_cfg.get("tasks", []):
        task_name = task_entry["task"]
        task_params = dict(task_entry.get("params", {}))
        for seed in seeds:
            print(f"[{task_name} seed={seed}] starting...", flush=True)
            try:
                metrics = run_single(
                    task_name=task_name,
                    task_params=task_params,
                    model_cfg=base_model,
                    training_cfg=base_training,
                    seed=seed,
                    discretization_threshold=threshold,
                    git_commit=git_commit,
                )
                runs.append({
                    "experiment_id": metrics["experiment_id"],
                    "task_name": task_name,
                    "task_params": task_params,
                    "seed": seed,
                    "status": "success",
                    "metrics": metrics,
                })
                print(f"[{task_name} seed={seed}] success "
                      f"disc_exact={metrics['discrete_exact_accuracy']:.4f} "
                      f"cont_exact={metrics['continuous_exact_accuracy']:.4f}",
                      flush=True)
            except Exception as exc:  # noqa: BLE001 - record and continue
                runs.append({
                    "experiment_id": None,
                    "task_name": task_name,
                    "task_params": task_params,
                    "seed": seed,
                    "status": "failed",
                    "error": f"{exc}\n{traceback.format_exc()[-2000:]}",
                })
                print(f"[{task_name} seed={seed}] FAILED: {exc}", flush=True)
    by_task: dict[str, list[dict]] = {}
    for r in runs:
        by_task.setdefault(r["task_name"], []).append(r)
    return {
        "suite_name": suite_cfg.get("suite_name", "suite"),
        "git_commit": git_commit,
        "total_runtime_seconds": time.time() - t0,
        "runs": runs,
        "summary": {t: summarize_task(rs) for t, rs in by_task.items()},
    }


def load_suite(name: str) -> dict:
    path = CONFIGS_DIR / f"{name}.json"
    if not path.exists():
        # Allow passing a direct path as --suite.
        path = Path(name)
    with open(path) as f:
        suite_cfg = json.load(f)
    # Shared model/training defaults live in baseline.json unless the
    # suite file defines its own.
    if "model" not in suite_cfg or "training" not in suite_cfg:
        with open(CONFIGS_DIR / "baseline.json") as f:
            baseline = json.load(f)
        suite_cfg.setdefault("model", baseline["model"])
        suite_cfg.setdefault("training", baseline["training"])
        suite_cfg.setdefault("discretization_threshold",
                             baseline.get("discretization_threshold", 0.5))
    return suite_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Boolean benchmark suite")
    parser.add_argument("--suite", default="baseline_suite")
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated seed list, overrides suite file")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    suite_cfg = load_suite(args.suite)
    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    else:
        seeds = [int(s) for s in suite_cfg.get("seeds", [0])]

    result = run_suite(suite_cfg, seeds, get_git_commit())
    print(json.dumps(result.get("summary", {}), indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
