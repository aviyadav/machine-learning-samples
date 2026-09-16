"""Option 6 -- side-step JIT entirely with ahead-of-time compilation.

``torch.compile`` is a *just-in-time* compiler, which is exactly why it can
surprise you: it compiles whatever shows up, whenever it shows up.  The
``torch.export`` path inverts that.  Shapes get bounded up front, the graph is
frozen, and ``torch._inductor.aoti_compile_and_package`` produces a standalone
package that can be loaded directly.

The trade is explicit:

* there is no recompilation *because there is no compiler at run time* -- no
  guards, no cache, no spikes;
* an input outside the declared bounds does not degrade, it **fails**, which is
  a property this module measures rather than assumes;
* bounding the shapes is real work, and applying Inductor code generation to the
  export gives up the target-agnosticism of ``torch.export`` itself.

So the run reports three things: the declared bounds, what actually happens when
an input crosses them, and how the AOT package performs against the same budget
as everything else.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

from recompilation_dilemma.config import RunConfig
from recompilation_dilemma.harness import Experiment, cache_root
from recompilation_dilemma.metrics import LatencyReport, Step
from recompilation_dilemma.solutions.base import SolutionOutcome

OPTION = 6
KEY = "aot-export"
NAME = "Fix the computation graph with ahead-of-time compilation"

#: Which axis of each input carries the request-dependent row count.
VARIABLE_DIMS: dict[str, int] = {
    "x": 0,
    "local_edge_index": 1,
    "cross_edge_index": 1,
    "local_edge_scale": 0,
    "cross_edge_scale": 0,
    "context": 0,
}


def build_dynamic_shapes(config: RunConfig) -> tuple:
    """Declare the bound of every dimension that varies between requests.

    Reusing one ``Dim`` object for the dimensions that must agree -- an edge
    tensor's axis and the matching per-edge scale -- makes that equality part of
    the exported contract instead of something the compiler has to guess, which
    is precisely the kind of guessing that produced the duck-shape guard in
    Option 3.
    """
    from torch.export import Dim

    upper = config.dataset.max_size
    nodes_dim = Dim("nodes", min=0, max=upper)
    local_edges_dim = Dim("local_edges", min=0, max=upper)
    cross_edges_dim = Dim("cross_edges", min=0, max=upper)
    context_dim = Dim("context", min=0, max=upper)

    return (
        {0: nodes_dim},
        {1: local_edges_dim},
        {1: cross_edges_dim},
        {0: local_edges_dim},
        {0: cross_edges_dim},
        {0: context_dim},
    )


def push_beyond_bounds(config: RunConfig, sample: dict[str, torch.Tensor]) -> dict:
    """Pad the variable axis of every input past the declared upper bound."""
    target = config.dataset.max_size + 1
    stretched = {}
    for name, tensor in sample.items():
        dim = VARIABLE_DIMS[name]
        current = tensor.shape[dim]
        if current >= target:
            stretched[name] = tensor
            continue
        padding_shape = list(tensor.shape)
        padding_shape[dim] = target - current
        padding = torch.zeros(padding_shape, dtype=tensor.dtype, device=tensor.device)
        stretched[name] = torch.cat([tensor, padding], dim=dim)
    return stretched


def probe_bounds(exported, config: RunConfig, sample: dict) -> str:
    """Feed the exported program an input it did not declare support for.

    This is the property that replaces the recompilation spike: instead of a
    multi-second stall deep inside the serving loop, an unsupported request
    produces an error at the boundary where it can be handled.
    """
    oversized = push_beyond_bounds(config, sample)
    try:
        with torch.inference_mode():
            exported.module()(*oversized.values())
    except Exception as error:  # noqa: BLE001 - the message is the finding
        first_line = str(error).splitlines()[0] if str(error) else type(error).__name__
        return f"rejected the request: {type(error).__name__}: {first_line}"
    return "ACCEPTED an input beyond the declared bounds (the boundary is not enforced)"


def measure_in_child(config: RunConfig) -> dict:
    """Package, load and serve the AOT model in a separate process.

    Returns the child's JSON payload.  A crash becomes ``{"ok": False, ...}``
    with the exit status, instead of taking this process down with it.
    """
    payload = json.dumps(
        {
            "device": config.device,
            "backend": config.backend,
            "model": {
                "feature_dim": config.model.feature_dim,
                "hidden_dim": config.model.hidden_dim,
                "message_passing_layers": config.model.message_passing_layers,
            },
            "dataset": {
                "samples": config.dataset.samples,
                "seed": config.dataset.seed,
                "min_size": config.dataset.min_size,
                "max_size": config.dataset.max_size,
            },
            "cache_dir": str(cache_root(config)),
        }
    )

    # The child imports this package by name, so make sure it is importable
    # regardless of the directory the CLI was launched from.
    package_root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(package_root), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    completed = subprocess.run(
        [sys.executable, "-m", "recompilation_dilemma.solutions._aot_child", payload],
        capture_output=True,
        text=True,
        env=environment,
    )

    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    detail = (completed.stderr or completed.stdout).strip().splitlines()
    return {
        "ok": False,
        "error": (
            f"child process exited with status {completed.returncode}"
            + (f": {detail[-1][:200]}" if detail else "")
        ),
    }


def run(experiment: Experiment, *, verbose: bool | None = None) -> SolutionOutcome:
    del verbose  # export emits no Dynamo recompilation logs
    config = experiment.config
    data = experiment.dataset
    model: nn.Module = experiment.fresh_model()

    exported = torch.export.export(
        model, tuple(data[0].values()), dynamic_shapes=build_dynamic_shapes(config)
    )
    bounds_report = probe_bounds(exported, config, data[0])
    bounds_note = f"boundary behaviour: {bounds_report}"

    result = measure_in_child(config)
    if not result.get("ok"):
        return SolutionOutcome(
            option=OPTION,
            key=KEY,
            name=NAME,
            report=None,
            tradeoff="removes the run-time compiler, and with it the possibility of a spike",
            caveat="shapes must be bounded up front and an out-of-bounds input fails",
            unavailable=(
                "the export and its bound checks succeeded, but packaging or "
                f"loading the AOT model failed here: {result.get('error', 'unknown')}"
            ),
            notes=[bounds_note],
        )

    timings: list[float] = result["timings"]
    steps = [
        # By construction there is no graph to compile and none to recompile.
        Step(index=index, ms=ms, compiled=True, new_graph=False)
        for index, ms in enumerate(timings)
    ]

    report = LatencyReport(
        label="Option 6: AOT export + Inductor package",
        mode="aot",
        steps=steps,
        # Nothing is compiled at run time, so there is no start-up compilation
        # to excuse: every request counts.
        cold_start=False,
        eager_baseline_ms=experiment.eager_ms,
        optimizing_backend=config.optimizing_backend,
        notes=[
            f"declared bounds: 0 <= rows/edges <= {config.dataset.max_size}",
            "no guards, no compilation cache, no recompilations by construction",
        ],
    )

    package_mb = result.get("package_bytes", 0) / (1024 * 1024)
    return SolutionOutcome(
        option=OPTION,
        key=KEY,
        name=NAME,
        report=report,
        tradeoff="removes the run-time compiler, and with it the possibility of a spike",
        caveat=(
            "requires bounded input dimensions; torch.export tends to be less "
            "flexible than torch.compile and may require code changes"
        ),
        notes=[
            bounds_note,
            f"packaged model: {package_mb:.1f} MiB",
        ],
    )
