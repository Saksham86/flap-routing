"""Seeding helpers. Call ``seed_everything`` once at the start of every script."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (CPU + all GPUs).

    ``deterministic=True`` additionally asks torch for deterministic kernels. It can slow
    things down and some ops raise if no deterministic implementation exists, so it is
    opt-in and recorded in the run metadata by the caller.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects subprocesses only
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # CPU-only analysis environments may not have torch
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
