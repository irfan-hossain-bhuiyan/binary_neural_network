"""Kaggle entry point for the I2 initialization reliability experiment."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,str(ROOT/'research'/'run_initialization_i2.py'),
                '--device','cuda','--epochs','3000',
                '--output',str(ROOT/'research'/'operator_results'/'initialization_i2_results.json')],check=True)
