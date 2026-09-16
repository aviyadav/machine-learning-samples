"""The PyTorch recompilation dilemma, expressed as runnable code.

The dilemma
-----------
``torch.compile`` makes inference faster on average, but every Dynamo guard
violation triggers a *recompilation*: a blocking, multi-second compilation that
lands in the middle of a serving loop.  For real-time inference that tail spike
is worse than not compiling at all, so teams fall back to eager execution and
give up the speedup.  That trade -- "fast on average but occasionally
catastrophic" versus "slow but bounded" -- is the dilemma.

This package contains both halves:

* :mod:`recompilation_dilemma.problem` reproduces the dilemma and states it as a
  machine-checkable budget (a tail bound *and* a minimum speedup that must both
  hold).
* :mod:`recompilation_dilemma.solutions` implements the six mitigations from the
  article and measures each one against that same budget.

Source article: "Overcoming the PyTorch Recompilation Dilemma in Real-Time
Inference Workloads" by Chaim Rand (Medium, Sep 2026).
"""

from __future__ import annotations

from recompilation_dilemma.config import LatencyBudgetSpec, ModelConfig, Profile, RunConfig
from recompilation_dilemma.dataset import INJECT_ID, make_dataset, warmup_dataset
from recompilation_dilemma.metrics import LatencyReport, Step, Verdict
from recompilation_dilemma.model import build_model

__all__ = [
    "INJECT_ID",
    "LatencyBudgetSpec",
    "LatencyReport",
    "ModelConfig",
    "Profile",
    "RunConfig",
    "Step",
    "Verdict",
    "build_model",
    "make_dataset",
    "warmup_dataset",
]

__version__ = "1.0.0"
