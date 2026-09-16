"""The problem: ``torch.compile`` is fast on average and catastrophic in the tail.

This module is the first half of the project.  It runs the two configurations a
serving team actually has on the table -- eager execution, and a naive
``torch.compile`` -- measures both against the same
:class:`~recompilation_dilemma.config.LatencyBudgetSpec`, and shows that neither
one is acceptable:

* **Eager** never spikes, but it never uses the compiled graph either, so it
  fails on compiled-path coverage (and on speedup too, wherever the hardware
  actually delivers one).
* **Naive ``torch.compile``** does use it, but every Dynamo guard violation
  blocks the serving loop on a full compilation, so it fails the ``max_step_ms``
  requirement -- by three to four orders of magnitude in the article's
  measurements.

The result is returned as a :class:`Dilemma` value whose ``demonstrated`` flag is
computed, not asserted in prose.  Every mitigation in
:mod:`recompilation_dilemma.solutions` exists to make *all* the checks pass at
once, and :mod:`recompilation_dilemma.cli` judges them against this same
predicate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from recompilation_dilemma.config import LatencyBudgetSpec
from recompilation_dilemma.harness import (
    Experiment,
    compile_model,
    dynamo_settings,
    log_recompiles,
    reset_compiler,
    run_compiled,
)
from recompilation_dilemma.metrics import LatencyReport, Step, Verdict


@dataclass
class Dilemma:
    """A measured statement of the trade-off, for one environment."""

    eager: LatencyReport
    compiled: LatencyReport
    budget: LatencyBudgetSpec
    notes: list[str] = field(default_factory=list)

    @property
    def eager_verdict(self) -> Verdict:
        return self.eager.verdict(self.budget)

    @property
    def compiled_verdict(self) -> Verdict:
        return self.compiled.verdict(self.budget)

    @property
    def demonstrated(self) -> bool | None:
        """Whether the environment reproduces the dilemma.

        ``None`` means "not evaluable here": with a speedup requirement set and a
        backend that generates no optimised code, the verdict is left unevaluated,
        so neither configuration can be called a failure.  Reporting ``None``
        rather than ``False`` keeps a smoke run with such a backend from looking
        like a disproof.  The check reads the verdicts rather than the backend
        name, so it stays correct when no speedup is demanded at all.
        """
        if self.eager_verdict.ok is None or self.compiled_verdict.ok is None:
            return None
        return self.eager_verdict.ok is False and self.compiled_verdict.ok is False

    @property
    def spike_multiple(self) -> float:
        return self.compiled.spike_multiple

    def describe(self) -> list[str]:
        lines = [
            f"eager      : mean {self.eager.serving_stats.mean_ms:.2f}ms"
            f", max {self.eager.serving_stats.max_ms:.2f}ms"
            f", compiled-path coverage {self.eager.compiled_ratio:.0%}"
            f"  -> {self.eager_verdict.symbol}",
            f"torch.compile: mean {self.compiled.steady_stats.mean_ms:.2f}ms"
            f" (steady), max {self.compiled.serving_stats.max_ms:.2f}ms"
            f", speedup {self.compiled.speedup_vs_eager:.2f}x"
            f"  -> {self.compiled_verdict.symbol}",
            f"tail spikes : {len(self.compiled.recompile_indices)} recompilations,"
            f" worst step is {self.spike_multiple:.0f}x the mean eager step",
        ]
        if self.demonstrated is True:
            lines.append(
                "dilemma reproduced: eager never uses the compiled graph, "
                "torch.compile misses the tail bound, and each fails a different "
                "part of the budget"
            )
        elif self.demonstrated is None:
            lines.append(
                "dilemma NOT evaluable here: a speedup is required but this backend "
                "generates no optimised code, so neither configuration is judged"
            )
        else:
            lines.append(
                "dilemma NOT reproduced: at least one configuration met the budget "
                "here"
            )
        lines.extend(f"note: {note}" for note in self.notes)
        return lines


def run(experiment: Experiment, *, verbose: bool | None = None) -> Dilemma:
    """Measure the eager baseline and a naive ``torch.compile`` run."""
    config = experiment.config
    verbose = config.verbose_recompiles if verbose is None else verbose

    eager_ms = experiment.eager_ms

    eager_report = LatencyReport(
        label="eager",
        mode="eager",
        # Eager never compiles, so there is no start-up compilation to excuse
        # and every request counts towards the baseline tail.
        cold_start=False,
        steps=[
            # The eager baseline has no graph and no spikes: it defines the
            # reference that the compiled run is measured against.
            Step(index=index, ms=ms, compiled=False, new_graph=False)
            for index, ms in enumerate(eager_ms)
        ],
        eager_baseline_ms=eager_ms,
        optimizing_backend=config.optimizing_backend,
    )

    reset_compiler(config)
    model = compile_model(experiment.fresh_model(), config)
    # The article raises the cache limit from its default of 8 to 64 for the
    # baseline run too, so the comparison is "compiled with enough room to
    # recompile" rather than "compiled until fullgraph gives up".
    with dynamo_settings(
        cache_size_limit=config.cache_size_limit
    ), log_recompiles(verbose):
        compiled_report = run_compiled(
            model,
            experiment.dataset,
            experiment.device,
            config=config,
            label="torch.compile (as-is)",
            mode=config.backend,
            eager_baseline_ms=eager_ms,
            notes=[
                "dynamic=True, fullgraph=True, "
                f"cache_size_limit={config.cache_size_limit}",
                "no mitigation applied",
            ],
        )

    return Dilemma(
        eager=eager_report,
        compiled=compiled_report,
        budget=config.budget,
    )
