"""Backward-compatible top-level module for the binary neural network.

The monolithic implementation has been split into focused submodules:

  - layers.py        : ``OrNorGateLayer`` and the continuous XOR helper.
  - initializers.py  : ``NormalInitWrapper``, ``ConstantInitWrapper``.
  - regularizers.py  : regularization factories.
  - models.py        : ``MultiLayerLogicGateNet``.
  - train_xor.py     : ``train_xor_main`` and ``main``.
  - parallel_xor.py  : parallel training driver.

This module re-exports the public symbols so existing imports keep working.
"""

from layers import xor, OrNorGateLayer
from initializers import NormalInitWrapper, ConstantInitWrapper
from regularizers import regularization_factory, regularization_factory2
from models import MultiLayerLogicGateNet
from train_xor import train_xor_main, main
from parallel_xor import run_train_xor_main_parallel

__all__ = [
    "xor",
    "OrNorGateLayer",
    "NormalInitWrapper",
    "ConstantInitWrapper",
    "regularization_factory",
    "regularization_factory2",
    "MultiLayerLogicGateNet",
    "train_xor_main",
    "main",
    "run_train_xor_main_parallel",
]

