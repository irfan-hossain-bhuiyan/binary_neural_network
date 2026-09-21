"""B3R seed-1 continuation from the verified v3 best-continuous state."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import shutil
import sys
import traceback
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import SigmoidOrModernLogicGateNet  # noqa: E402
from research.boolean_tasks import build_task  # noqa: E402
from research.run_or_b3 import evaluate, gate_stats, lehmer_gradient_report  # noqa: E402

BASE = Path("artifacts/checkpoints/B3R_v3/B3R_lehmer_p2_seed1_best_continuous_mse.pt")
EXPORT = Path("artifacts/checkpoints/B3R_continuation")


def make_model(device):
    return SigmoidOrModernLogicGateNet(
        8, 4, width=64, num_residual_blocks=2, or_operator="lehmer_p2",
        bias_initialization=lambda t: nn.init.normal_(t, mean=.5, std=.1),
    ).to(device)


def main() -> None:
    try:
        seed = 1
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        task = build_task("bitwise_xor_truth_table", {"bits": 4})
        x, y = task["X"].float().to(device), task["Y"].float().to(device)
        model = make_model(device)
        base_hash = hashlib.sha256(BASE.read_bytes()).hexdigest()
        model.load_state_dict(torch.load(BASE, map_location=device, weights_only=True))
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=.01)
        initial = evaluate(model, x, y)
        best = None; best_boolean = None; trajectory = []
        shutil.rmtree(EXPORT, ignore_errors=True)
        EXPORT.mkdir(parents=True, exist_ok=True)
        manifest = []

        def save_verified(kind, step, record):
            # Best-state updates can happen many times.  Include the step in
            # every filename so manifest entries always refer to immutable
            # bytes rather than an overwritten path.
            path = EXPORT / f"B3R_CONT_lehmer_p2_seed1_{kind}_step{step}.pt"
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
            check = make_model(device)
            check.load_state_dict(torch.load(path, map_location=device, weights_only=True)); check.eval()
            actual = evaluate(check, x, y)
            for mode in ("continuous", "hard", "boolean"):
                for metric in ("mse", "bit_accuracy", "exact_accuracy"):
                    if abs(actual[mode][metric] - record[mode][metric]) > 1e-6:
                        raise RuntimeError(f"reload verification failed: {path} {mode}.{metric}")
            manifest.append({"checkpoint": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "step": step, "kind": kind, "metrics": actual})

        for step in range(1, 3001):
            out = model(x); loss = (out - y).square().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 25 != 0 and step != 3000:
                continue
            model.eval(); ev = evaluate(model, x, y)
            rec = {"step": step, **ev, "gate_stats": gate_stats(model), "lehmer_gradient_report": lehmer_gradient_report(model, x)}
            trajectory.append(rec)
            mse = ev["continuous"]["mse"]
            if best is None or mse < best["continuous"]["mse"]:
                best = rec; model.train()
                save_verified("best_continuous_mse", step, rec)
            if best_boolean is None or ev["boolean"]["exact_accuracy"] > best_boolean["boolean"]["exact_accuracy"]:
                best_boolean = rec; model.train()
                save_verified("best_boolean_exact", step, rec)
            if step in (500, 1000, 1500, 2000, 2500, 3000):
                model.train(); save_verified(f"step{step}", step, rec)
            model.train()
        model.eval(); final = evaluate(model, x, y)
        save_verified("final", 3000, final)
        payload = {
            "experiment": "B3R-seed1-continuation",
            "parent_experiment": "B3R-checkpoint-export-replication",
            "parent_checkpoint": str(BASE), "parent_checkpoint_sha256": base_hash,
            "operator": "lehmer_p2", "seed": 1, "additional_steps": 3000,
            "device": str(device), "architecture": {"input": 8, "width": 64, "blocks": 2, "output": 4},
            "training": {"optimizer": "Adam", "lr": .01, "loss": "MSE", "batch_size": 256, "historical_inclusive_loop": True},
            "runtime": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark},
            "initial": initial, "best": best, "best_boolean": best_boolean, "final": final,
            "trajectory": trajectory, "checkpoint_manifest": manifest,
        }
        Path("b3r_continuation_manifest.json").write_text(json.dumps({"experiment": "B3R-seed1-continuation", "checkpoints": manifest}, indent=2))
        Path("b3r_continuation_metrics.json").write_text(json.dumps(payload, indent=2))
        Path("/kaggle/working/result.json").write_text(json.dumps(payload, indent=2))
    except Exception:
        Path("b3r_continuation_error.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
