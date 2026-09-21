"""Kaggle entry point for the full replicated I1R diagnostic."""
from __future__ import annotations
import shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
result = ROOT / "research" / "operator_results" / "initialization_i1r_full.json"
subprocess.run([
    sys.executable, str(ROOT / "research" / "analyze_meanfield_i1r.py"),
    "--network-seeds", "64",
    "--residual-seeds", "64",
    "--bool-batch", "8192",
    "--continuous-batch", "512",
    "--device", "cuda",
    "--output", str(result),
], check=True)
print(result)
