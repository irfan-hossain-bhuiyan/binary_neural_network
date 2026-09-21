"""Kaggle entry point for I2F's four missing initialization cells."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
result = ROOT / "research" / "operator_results" / "initialization_i2f_results.json"
subprocess.run([
    sys.executable,
    str(ROOT / "research" / "run_initialization_i2f.py"),
    "--device", "cuda",
    "--epochs", "3000",
], check=True)
print(result)
