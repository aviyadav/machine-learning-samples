"""Option 5 -- run the recompilation on a background thread.

The other mitigations avoid or schedule the compilation.  This one keeps the
compilation exactly where it was and moves the *blocking*: a background worker
takes the sample that missed its guards, and the foreground serving thread
immediately falls back to eager execution instead of waiting.  When the worker
finishes, the foreground quietly returns to the compiled model.

Two things make this hard, and both are properties of Python rather than of
PyTorch:

* **The GIL.**  A compilation is long and CPU-bound.  A background thread doing
  it holds the GIL in long slices, starving the foreground thread that is trying
  to serve the fallback eagerly.  The worst case observed in the article is a
  fallback step that takes 1,580ms instead of the ~65ms it should.

* **Process boundaries.**  The Dynamo guards that the worker extends live in
  process memory.  That is why the worker is a thread and not a process -- a
  separate process would compile a graph the serving process could not use.

Because of the first point, the levers that matter are the ones that limit the
worker's hold on the CPU: CPU affinity, thread priority, the compiler's own
thread count, and ``sys.setswitchinterval``.  :class:`InterferenceControls`
records which of them this platform actually honoured, so the result is never
read as better than the environment allows.
"""

from __future__ import annotations

import contextlib
import queue
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch._dynamo as dynamo
import torch.nn as nn

from recompilation_dilemma.config import RunConfig
from recompilation_dilemma.harness import (
    Experiment,
    dynamo_settings,
    log_recompiles,
    reset_compiler,
)
from recompilation_dilemma.metrics import LatencyReport, Step
from recompilation_dilemma.runtime import (
    InterferenceControls,
    StepTimer,
    lower_current_thread_priority,
    pin_current_thread_to_one_cpu,
)
from recompilation_dilemma.solutions.base import SolutionOutcome, pick_best

OPTION = 5
KEY = "async-background"
NAME = "Execute recompilations on a designated background thread"

#: The GIL switch intervals the article compares: stock, then 100 microseconds.
DEFAULT_GIL_INTERVALS: tuple[float | None, ...] = (None, 0.1)

#: How a request was served.
COMPILED = "compiled"
DISPATCHED = "dispatched"
DEFERRED = "deferred"


@dataclass
class AsyncModelStats:
    compiled_runs: int = 0
    dispatched_runs: int = 0
    deferred_runs: int = 0
    completed_compilations: int = 0


class AsyncCompileWrapper:
    """A model wrapper for asynchronous background recompilations."""

    def __init__(
        self,
        model: nn.Module,
        config: RunConfig,
        *,
        gil_interval_ms: float | None = None,
    ) -> None:
        self.eager = model.eval()
        self.compiled = torch.compile(
            model, backend=config.backend, fullgraph=True, dynamic=True
        )
        self.gil_interval_ms = gil_interval_ms
        self.jobs: queue.Queue = queue.Queue(maxsize=1)

        # Whether the worker currently owns a compilation job.  While it is set,
        # the foreground must not touch the compiled model.
        self.compiling = threading.Event()
        self.completed = 0
        self.stats = AsyncModelStats()
        self.controls = InterferenceControls()

        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    @contextlib.contextmanager
    def _recompile_error(self, enabled: bool) -> Iterator[None]:
        """The foreground enables this so a guard miss raises; the worker
        disables it so the same miss compiles normally."""
        previous = dynamo.config.error_on_recompile
        dynamo.config.error_on_recompile = enabled
        try:
            yield
        finally:
            dynamo.config.error_on_recompile = previous

    def _worker(self) -> None:
        # Pin this thread to a single CPU where the platform supports it.
        previous_affinity = pin_current_thread_to_one_cpu()
        self.controls.pinned_cpu = bool(previous_affinity)
        self.controls.lowered_priority = lower_current_thread_priority(10)
        if not self.controls.lowered_priority:
            self.controls.notes.append(
                "thread priority could not be lowered on this platform"
            )

        try:
            import torch._inductor.config as inductor_config

            inductor_config.compile_threads = 1
            self.controls.single_compile_thread = True
        except (ImportError, AttributeError) as error:
            self.controls.notes.append(f"compile_threads unavailable: {error}")

        while (sample := self.jobs.get()) is not None:
            previous_interval = sys.getswitchinterval()
            if self.gil_interval_ms is not None:
                sys.setswitchinterval(self.gil_interval_ms / 1_000)
            try:
                with torch.inference_mode(), self._recompile_error(False):
                    self.compiled(**sample)
                self.completed += 1
            finally:
                # The foreground may try the compiled model again.
                self.compiling.clear()
                sys.setswitchinterval(previous_interval)
                self.jobs.task_done()

        # Release the sentinel item so that jobs.join() can return.
        self.jobs.task_done()

    def __call__(self, sample: dict[str, torch.Tensor]) -> tuple[torch.Tensor, str]:
        if self.compiling.is_set():
            # A worker already owns a compilation job, so serve this request
            # eagerly rather than queueing behind it.
            self.stats.deferred_runs += 1
            return self.eager(**sample), DEFERRED

        try:
            # The following raises on a guard miss.
            with self._recompile_error(True):
                output = self.compiled(**sample)
            self.stats.compiled_runs += 1
            return output, COMPILED
        except dynamo.exc.RecompileError:
            # Compile this triggering sample on the background thread, and serve
            # it eagerly now.
            self.compiling.set()
            self.jobs.put(sample)
            self.stats.dispatched_runs += 1
            return self.eager(**sample), DISPATCHED

    def close(self) -> None:
        self.jobs.put(None)
        self.jobs.join()
        self.worker.join(timeout=5)
        self.stats.completed_compilations = self.completed


def run_async(
    model: AsyncCompileWrapper,
    data: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> list[Step]:
    """Serve the dataset through the wrapper, recording how each step was served."""
    steps: list[Step] = []
    for index, sample in enumerate(data):
        with StepTimer(device) as timer, torch.inference_mode():
            _, status = model(sample)
        steps.append(
            Step(
                index=index,
                ms=timer.elapsed_ms,
                compiled=status is COMPILED,
                # A dispatched request is one that would have blocked the
                # serving loop on a compilation; those are the spikes this
                # mitigation removes.
                new_graph=status is DISPATCHED,
            )
        )
    return steps


def build_report(
    model: AsyncCompileWrapper,
    steps: list[Step],
    *,
    config: RunConfig,
    label: str,
    eager_baseline_ms: list[float],
) -> LatencyReport:
    """Summarise a run of the wrapper.  Call *after* ``model.close()``.

    The background-completion count only becomes final once the worker has been
    joined, so building the report earlier would always report zero compilations
    completed -- the one number that tells you whether the worker ever caught up
    with the serving loop.
    """
    return LatencyReport(
        label=label,
        mode=f"{config.backend}+async",
        steps=steps,
        eager_baseline_ms=eager_baseline_ms,
        optimizing_backend=config.optimizing_backend,
        notes=[
            f"served compiled: {model.stats.compiled_runs}, "
            f"dispatched to worker: {model.stats.dispatched_runs}, "
            f"deferred while worker busy: {model.stats.deferred_runs}",
            f"background compilations completed: {model.stats.completed_compilations}",
            f"interference controls honoured: {model.controls.describe()}",
            *model.controls.notes,
        ],
    )


def _variant(
    config: RunConfig,
    experiment: Experiment,
    gil_interval_ms: float | None,
    *,
    verbose: bool,
) -> LatencyReport:
    reset_compiler(config)
    wrapper = AsyncCompileWrapper(
        experiment.fresh_model(), config, gil_interval_ms=gil_interval_ms
    )
    try:
        # Without a raised limit the worker eventually trips Dynamo's default
        # recompile_limit, which -- because the wrapper compiles with
        # fullgraph=True -- is a hard failure rather than a fallback.
        with dynamo_settings(
            cache_size_limit=config.cache_size_limit
        ), log_recompiles(verbose):
            steps = run_async(wrapper, experiment.dataset, experiment.device)
    finally:
        # Joins the worker, which is what finalises the completion count.
        wrapper.close()

    return build_report(
        wrapper,
        steps,
        config=config,
        label=f"Option 5: async background compile (gil={gil_interval_ms})",
        eager_baseline_ms=experiment.eager_ms,
    )


def run(
    experiment: Experiment,
    *,
    gil_intervals: tuple[float | None, ...] = DEFAULT_GIL_INTERVALS,
    verbose: bool | None = None,
) -> SolutionOutcome:
    config = experiment.config
    verbose = config.verbose_recompiles if verbose is None else verbose

    variants = [
        _variant(config, experiment, interval, verbose=verbose)
        for interval in gil_intervals
    ]
    best = pick_best(variants, config.budget)

    notes = []
    for interval, report in zip(gil_intervals, variants, strict=True):
        # Only requests that were actually served eagerly while a compilation was
        # in flight.  The article reports the same two numbers, because tuning the
        # GIL interval is about how much the background compile disturbs them.
        fallback_ms = [
            step.ms for step in report.steps if not step.compiled and step.index >= 1
        ]
        if fallback_ms:
            disturbance = (
                f"eager-served mean {sum(fallback_ms) / len(fallback_ms):.2f}ms, "
                f"max {max(fallback_ms):.2f}ms"
            )
        else:
            disturbance = "no request was served eagerly; nothing was dispatched"
        notes.append(
            f"gil_interval_ms={interval}: {disturbance}, "
            f"overall max {report.serving_stats.max_ms:.2f}ms"
        )

    # The whole technique rests on compilations being rare relative to traffic,
    # so say plainly when they were not.  On a CPU one compilation can outlast
    # the entire serving loop, and the wrapper then degrades to eager for every
    # request after the first -- which is the caveat made concrete rather than a
    # measurement failure.
    if best.compiled_ratio < config.budget.min_compiled_ratio:
        notes.append(
            "the serving loop finished before the background compilation did, so "
            "the wrapper fell back to eager for almost every request: this "
            "workload does not have enough traffic per compilation for the "
            "technique to pay off"
        )

    return SolutionOutcome(
        option=OPTION,
        key=KEY,
        name=NAME,
        report=best,
        tradeoff="removes the blocking wait; the work still happens off the request path",
        caveat=(
            "needs spare CPU/GPU capacity and a GIL interval tuned to keep the "
            "background compile from starving the eager fallback"
        ),
        variants=variants,
        notes=notes,
    )
