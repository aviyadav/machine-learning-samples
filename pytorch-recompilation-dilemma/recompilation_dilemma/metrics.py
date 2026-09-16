"""Latency accounting, and the budget that turns the dilemma into a boolean.

The whole point of this module is :class:`LatencyBudget` plus
:meth:`LatencyReport.verdict`.  A mitigation has to satisfy two constraints at
once -- a tail bound *and* a minimum speedup -- and that conjunction is what
makes the recompilation dilemma a real dilemma rather than a tuning exercise:

* eager execution satisfies the tail bound and fails the speedup requirement;
* naive ``torch.compile`` satisfies the speedup requirement and fails the tail
  bound.

Every experiment in this project is judged by the same predicate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from recompilation_dilemma.config import LatencyBudgetSpec


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile of ``values`` (``q`` in ``[0, 100]``)."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class StepStats:
    """Descriptive statistics for a set of per-step latencies."""

    count: int
    mean_ms: float
    p50_ms: float
    p99_ms: float
    max_ms: float

    @classmethod
    def of(cls, values: list[float]) -> StepStats:
        return cls(
            count=len(values),
            mean_ms=sum(values) / len(values),
            p50_ms=percentile(values, 50),
            p99_ms=percentile(values, 99),
            max_ms=max(values),
        )


@dataclass(frozen=True)
class Step:
    """One inference request."""

    index: int
    ms: float
    compiled: bool
    """False when the request was served eagerly -- either the baseline run, or
    a guard-miss fallback under a hybrid scheme."""

    new_graph: bool
    """True when serving this request caused Dynamo to build a new graph."""


@dataclass
class LatencyReport:
    """Everything measured about one serving configuration."""

    label: str
    mode: str
    steps: list[Step]
    eager_baseline_ms: list[float] | None = None
    optimizing_backend: bool = True
    cold_start: bool = True
    """Whether the compiler had no graphs at all when serving began.

    Only then is request 0 excused as one-off start-up cost.  After a warm-up
    phase or a preloaded session the compiler is already primed, so request 0 is
    an ordinary request and its latency counts -- otherwise a mitigation that
    fails on the very first request would be scored as if nothing had happened.
    """
    notes: list[str] = field(default_factory=list)

    # -- derived views ----------------------------------------------------

    @property
    def timings(self) -> list[float]:
        return [step.ms for step in self.steps]

    @property
    def recompile_indices(self) -> list[int]:
        """Sample indices whose serving blocked on a fresh compilation."""
        return [step.index for step in self.steps if step.new_graph]

    @property
    def fallback_indices(self) -> list[int]:
        """Sample indices served eagerly instead of by the compiled graph."""
        return [step.index for step in self.steps if not step.compiled]

    @property
    def steady_indices(self) -> list[int]:
        """Requests on the warm compiled path: no setup, no recompile spikes."""
        spikes = set(self.recompile_indices)
        return [
            step.index
            for step in self.steps
            if self.is_served(step.index) and step.index not in spikes
        ]

    @property
    def steady_stats(self) -> StepStats:
        """Latency of the warm compiled path -- the speedup that is on offer."""
        values = [self.steps[i].ms for i in self.steady_indices]
        if not values:
            values = [step.ms for step in self.steps if step.index >= 1] or self.timings
        return StepStats.of(values)

    @property
    def spike_stats(self) -> StepStats | None:
        values = [self.steps[i].ms for i in self.recompile_indices]
        return StepStats.of(values) if values else None

    @property
    def serving_stats(self) -> StepStats:
        """Every request a caller experiences -- the real tail."""
        values = [
            step.ms for step in self.steps if self.is_served(step.index)
        ]
        return StepStats.of(values or self.timings)

    def is_served(self, index: int) -> bool:
        """Whether request ``index`` counts towards serving latency."""
        return index >= 1 or not self.cold_start

    @property
    def speedup_vs_eager(self) -> float:
        """Steady-state speedup over the eager baseline.

        Measured the way the article measures it: the sum of eager step times
        divided by the sum of compiled step times, over the warm, non-spiking
        requests only.
        """
        if self.eager_baseline_ms is None:
            return 1.0
        indices = self.steady_indices
        if not indices:
            return 1.0
        eager_total = sum(self.eager_baseline_ms[i] for i in indices)
        served_total = sum(self.steps[i].ms for i in indices)
        return eager_total / served_total if served_total else float("inf")

    @property
    def eager_mean_ms(self) -> float:
        if self.eager_baseline_ms is None:
            return self.serving_stats.mean_ms
        return sum(self.eager_baseline_ms) / len(self.eager_baseline_ms)

    @property
    def spike_multiple(self) -> float:
        """Worst step as a multiple of the average eager step."""
        baseline = self.eager_mean_ms
        return self.serving_stats.max_ms / baseline if baseline else float("inf")

    def counts_over(self, max_step_ms: float) -> int:
        """Requests past ``max_step_ms``, on the same terms as ``serving_stats``."""
        return sum(
            1
            for step in self.steps
            if self.is_served(step.index) and step.ms > max_step_ms
        )

    @property
    def compiled_ratio(self) -> float:
        """Fraction of served requests that actually used the compiled graph.

        This is the check that keeps the budget honest: a configuration can pass
        the tail bound trivially by not compiling anything, and the speedup is
        only worth having if requests are on the compiled path in the first
        place.  It is also exactly what a cache-size limit trades away when it is
        set too low.
        """
        served = [step for step in self.steps if self.is_served(step.index)]
        if not served:
            return 0.0
        return sum(1 for step in served if step.compiled) / len(served)

    def verdict(self, budget: LatencyBudgetSpec) -> Verdict:
        return Verdict.of(self, budget)

    # -- presentation ----------------------------------------------------

    def summary_lines(self) -> list[str]:
        lines = [
            f"mode={self.mode}",
            f"recompilations: {len(self.recompile_indices)} {self.recompile_indices}",
            f"compiled runs: {len(self.steps) - len(self.fallback_indices)}"
            f", eager fallbacks: {len(self.fallback_indices)}",
            f"steady mean: {self.steady_stats.mean_ms:.2f}ms"
            f"  (p99 {self.steady_stats.p99_ms:.2f}ms)",
            f"serving max: {self.serving_stats.max_ms:.2f}ms",
        ]
        if self.eager_baseline_ms is not None:
            lines.append(
                f"speedup vs eager (steady): {self.speedup_vs_eager:.2f}x"
                f"  [eager mean {self.eager_mean_ms:.2f}ms]"
            )
        spikes = self.spike_stats
        if spikes is not None:
            lines.extend(
                [
                    f"recompile mean: {spikes.mean_ms:.2f}ms",
                    f"recompile max: {spikes.max_ms:.2f}ms",
                    f"worst step / eager mean: {self.spike_multiple:.0f}x",
                ]
            )
        lines.extend(f"note: {note}" for note in self.notes)
        return lines


@dataclass(frozen=True)
class Verdict:
    """The budget check, as a value the CLI and any caller can both read."""

    label: str
    tail_ok: bool
    max_step_ms: float
    observed_max_ms: float
    coverage_ok: bool
    compiled_ratio: float
    min_compiled_ratio: float
    speedup_ok: bool | None
    speedup: float
    min_speedup: float | None
    ok: bool | None
    notes: list[str] = field(default_factory=list)

    @classmethod
    def of(cls, report: LatencyReport, budget: LatencyBudgetSpec) -> Verdict:
        observed_max = report.serving_stats.max_ms
        tail_ok = observed_max <= budget.max_step_ms

        compiled_ratio = report.compiled_ratio
        coverage_ok = compiled_ratio >= budget.min_compiled_ratio

        speedup = report.speedup_vs_eager
        notes: list[str] = []

        if not tail_ok:
            notes.append(
                f"{report.counts_over(budget.max_step_ms)} request(s) exceeded the "
                f"{budget.max_step_ms:.0f}ms bound; worst was {observed_max:.0f}ms"
            )
        if not coverage_ok:
            notes.append(
                f"only {compiled_ratio:.0%} of served requests used the compiled "
                f"graph, below the required {budget.min_compiled_ratio:.0%} -- the "
                "speedup is not actually in play"
            )

        if budget.min_speedup is None:
            # No speedup is being demanded, so there is nothing to evaluate.
            speedup_ok: bool | None = None
            ok: bool | None = tail_ok and coverage_ok
            notes.append(
                "no steady-state speedup is required in this profile, so only "
                "the tail bound and compiled-path coverage are enforced"
            )
        elif not report.optimizing_backend:
            speedup_ok = None
            ok = None
            notes.append(
                f"backend={report.mode!r} does not generate optimised code, so the "
                f"{budget.min_speedup:.2f}x speedup requirement is not evaluable; "
                "only the recompilation machinery is being exercised"
            )
        else:
            speedup_ok = speedup >= budget.min_speedup
            ok = tail_ok and coverage_ok and speedup_ok
            if not speedup_ok:
                notes.append(
                    f"steady speedup {speedup:.2f}x is below the required "
                    f"{budget.min_speedup:.2f}x"
                )

        return cls(
            label=report.label,
            tail_ok=tail_ok,
            max_step_ms=budget.max_step_ms,
            observed_max_ms=observed_max,
            coverage_ok=coverage_ok,
            compiled_ratio=compiled_ratio,
            min_compiled_ratio=budget.min_compiled_ratio,
            speedup_ok=speedup_ok,
            speedup=speedup,
            min_speedup=budget.min_speedup,
            ok=ok,
            notes=notes,
        )

    @property
    def symbol(self) -> str:
        if self.ok is None:
            return "n/a"
        return "PASS" if self.ok else "FAIL"
