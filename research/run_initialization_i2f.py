"""I2F: missing 3x3 initialization-factorial cells.

This runner reuses the historical I2 training loop and diagnostics, but
trains only the four cells absent from the original I2 run.  Existing I2
artifacts are never touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import run_initialization_i2 as base


OUT = ROOT / "research" / "operator_results"
CHECKPOINTS = OUT / "initialization_i2f_checkpoints"
EXISTING_I2 = OUT / "initialization_i2_results.json"

CONDITIONS = {
    "I2-F": {"name": "historical_one", "edge": "historical", "bias": "ONE"},
    "I2-G": {"name": "historical_balanced", "edge": "historical", "bias": "BALANCED_POLARIZED"},
    "I2-H": {"name": "mf_sigma2_current", "edge": "meanfield", "sigma": 2.0, "bias": "CURRENT"},
    "I2-I": {"name": "mf_sigma4_current", "edge": "meanfield", "sigma": 4.0, "bias": "CURRENT"},
}


def paired_bias_init(name: str, seed: int):
    """Gaussian bias initializer with deterministic tie handling.

    CURRENT and BALANCED_POLARIZED share the same standard-normal draws.  A
    vanishingly rare float32 round-to-0.5 tie is nudged toward the sampled
    sign so the intended paired threshold mask is preserved exactly.
    """
    mean, std = base.BIAS[name]
    generator = base.gen(seed)

    def init(tensor: torch.Tensor) -> None:
        with torch.no_grad():
            z = torch.empty_like(tensor).normal_(0.0, 1.0, generator=generator)
            raw = mean + std * z
            if mean == 0.5:
                tie = (raw == 0.5) & (z != 0)
                up = torch.nextafter(torch.tensor(0.5, device=tensor.device, dtype=tensor.dtype),
                                     torch.tensor(float("inf"), device=tensor.device, dtype=tensor.dtype))
                down = torch.nextafter(torch.tensor(0.5, device=tensor.device, dtype=tensor.dtype),
                                       torch.tensor(float("-inf"), device=tensor.device, dtype=tensor.dtype))
                raw = torch.where(tie & (z > 0), up, raw)
                raw = torch.where(tie & (z < 0), down, raw)
            tensor.copy_(raw)

    return init


def paired_check(condition: str, seed: int, device: torch.device) -> None:
    """Assert pairing against the archived I2 initial masks and new cells."""
    old = json.loads(EXISTING_I2.read_text())
    rows = {(row["condition"], int(row["seed"])): row for row in old["results"]}

    model = base.model_for(condition, seed, device)
    edge_hash = base.mask_hash(model)
    bias_hash = base.mask_hash(model, True)

    if condition in ("I2-F", "I2-G"):
        other = base.model_for("I2-G" if condition == "I2-F" else "I2-F", seed, device)
        assert edge_hash == base.mask_hash(other), f"historical edge pairing failed seed={seed}"
        assert edge_hash == rows[("I2-A", seed)]["initialization"]["edge_mask_sha256"]
        if condition == "I2-G":
            # BALANCED_POLARIZED is paired with a CURRENT Gaussian probe
            # using the same standard-normal draws.  I2-A intentionally
            # retains the historical global-RNG CURRENT stream.
            paired_current = base.SigmoidOrModernLogicGateNet(
                8, 4, width=64, num_residual_blocks=2, or_operator="lehmer_p2",
                gate_initializations=[.75, .25, .75, .25, .75, .25],
                bias_initialization=base.bias_init("CURRENT", 200000 + seed),
            ).to(device)
            assert bias_hash == base.mask_hash(paired_current, True), f"balanced/current bias pairing failed seed={seed}"
    elif condition == "I2-H":
        assert edge_hash == rows[("I2-B", seed)]["initialization"]["edge_mask_sha256"]
        assert edge_hash == rows[("I2-D", seed)]["initialization"]["edge_mask_sha256"]
        paired = base.SigmoidOrModernLogicGateNet(
            8, 4, width=64, num_residual_blocks=2, or_operator="lehmer_p2",
            gate_initializations=[.5] * 6,
            bias_initialization=paired_bias_init("BALANCED_POLARIZED", 200000 + seed),
            edge_initialization=base.edge_init(2.0, 100000 + seed),
        ).to(device)
        assert bias_hash == base.mask_hash(paired, True), f"MF sigma2 bias stream mismatch seed={seed}"
    elif condition == "I2-I":
        assert edge_hash == rows[("I2-C", seed)]["initialization"]["edge_mask_sha256"]
        assert edge_hash == rows[("I2-E", seed)]["initialization"]["edge_mask_sha256"]
        paired = base.SigmoidOrModernLogicGateNet(
            8, 4, width=64, num_residual_blocks=2, or_operator="lehmer_p2",
            gate_initializations=[.5] * 6,
            bias_initialization=paired_bias_init("BALANCED_POLARIZED", 200000 + seed),
            edge_initialization=base.edge_init(4.0, 100000 + seed),
        ).to(device)
        assert bias_hash == base.mask_hash(paired, True), f"MF sigma4 bias stream mismatch seed={seed}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not EXISTING_I2.exists():
        raise FileNotFoundError(EXISTING_I2)
    if CHECKPOINTS.exists() and any(CHECKPOINTS.iterdir()):
        raise RuntimeError(f"refusing to reuse non-empty I2F checkpoint directory: {CHECKPOINTS}")
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    base.CONDITIONS = CONDITIONS
    base.bias_init = paired_bias_init
    results, manifest = base.run(
        args.epochs,
        range(5),
        device,
        conditions=CONDITIONS,
        checkpoint_root=CHECKPOINTS,
        prefix="I2F",
        paired_check=paired_check,
    )
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload = {
        "experiment": "I2F-missing-initialization-factorial",
        "git_sha": sha,
        "device": str(device),
        "epochs": args.epochs,
        "optimizer_steps": args.epochs + 1,
        "conditions": CONDITIONS,
        "seeds": list(range(5)),
        "training": {
            "optimizer": "Adam", "lr": 0.01, "loss": "MSE", "batch_size": 256,
            "regularizer": None, "noise": None, "scheduler": None, "operator": "lehmer_p2",
        },
        "results": results,
        "checkpoint_manifest": manifest,
    }
    output = OUT / "initialization_i2f_results.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    manifest_path = OUT / "initialization_i2f_checkpoint_manifest.json"
    manifest_path.write_text(json.dumps({"experiment": "I2F", "git_sha": sha, "checkpoints": manifest}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
