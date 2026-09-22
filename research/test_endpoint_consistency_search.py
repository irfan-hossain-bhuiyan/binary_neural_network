"""Small numerical search for finite-error endpoint counterexamples.

This is deliberately a diagnostic, not a proof: it samples tiny Boolean
circuits and reports any case where a very small continuous error coexists
with a different thresholded circuit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.or_surrogates import get_operator


def lehmer(v: torch.Tensor) -> torch.Tensor:
    return get_operator("lehmer_p2")(v)


def search(seed: int, trials: int, threshold: float, tol: float) -> dict:
    gen = torch.Generator().manual_seed(seed)
    found = []
    # Direct one-neuron clauses cover the endpoint argument without hiding it
    # behind a full network optimizer.
    for n in (1, 2, 4):
        literals = torch.randint(0, 2, (trials, n), generator=gen, dtype=torch.float64)
        logits = torch.randn((trials, n), generator=gen, dtype=torch.float64)
        gates = torch.sigmoid(logits)
        values = literals * gates
        out = lehmer(values)
        target = literals.max(dim=-1).values
        discrete = (literals * (gates >= threshold)).max(dim=-1).values
        bad = (out - target).abs() < tol
        bad &= discrete != target
        for i in bad.nonzero(as_tuple=False).flatten().tolist()[:10]:
            found.append({"fan_in": n, "continuous_error": float((out[i] - target[i]).abs()),
                          "target": float(target[i]), "discrete": float(discrete[i]),
                          "literal": literals[i].tolist(), "gate": gates[i].tolist()})
    return {"seed": seed, "trials_per_fan_in": trials, "threshold": threshold,
            "tolerance": tol, "counterexamples": found}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trials", type=int, default=100_000)
    p.add_argument("--threshold", type=float, default=.5)
    p.add_argument("--tolerance", type=float, default=1e-10)
    p.add_argument("--output", default="")
    args = p.parse_args()
    result = search(args.seed, args.trials, args.threshold, args.tolerance)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
