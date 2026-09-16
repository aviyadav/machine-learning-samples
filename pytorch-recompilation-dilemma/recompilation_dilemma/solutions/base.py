"""Shared types for the six mitigations.

A mitigation is a function ``(Experiment) -> SolutionOutcome``.  The outcome
carries more than a latency report: every one of the six trades something away,
and the project is only honest if the code says what.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from recompilation_dilemma.config import LatencyBudgetSpec
from recompilation_dilemma.harness import Experiment
from recompilation_dilemma.metrics import LatencyReport, Verdict


@dataclass
class SolutionOutcome:
    """What one mitigation achieved, and what it cost."""

    option: int
    key: str
    name: str
    report: LatencyReport | None
    tradeoff: str
    """What this buys, stated as a mechanism rather than a benefit."""
    caveat: str
    """The condition under which the mitigation stops working."""
    variants: list[LatencyReport] = field(default_factory=list)
    """Sub-configurations that were measured, e.g. each cache limit tried."""
    notes: list[str] = field(default_factory=list)
    unavailable: str | None = None
    """Set when the mitigation could not run in this environment at all."""

    @property
    def ran(self) -> bool:
        return self.report is not None and self.unavailable is None

    def verdict(self, budget: LatencyBudgetSpec) -> Verdict | None:
        return self.report.verdict(budget) if self.report else None

    def headline(self) -> str:
        if not self.ran:
            # The reason can be a long exception message; the detail follows in the
            # outcome's notes, so the banner stays readable.
            reason = (self.unavailable or "unknown reason").strip()
            if len(reason) > 110:
                reason = reason[:107].rstrip() + "..."
            return f"Option {self.option}: {self.name} -- not run ({reason})"
        assert self.report is not None
        return f"Option {self.option}: {self.name} -- {self.report.label}"


#: Signature every mitigation module exposes as ``run``.
Mitigation = Callable[..., SolutionOutcome]


def pick_best(
    reports: list[LatencyReport], budget: LatencyBudgetSpec
) -> LatencyReport:
    """Choose the variant a serving team would actually deploy.

    Preference order: smallest worst-case step (the whole point is the tail),
    then best steady-state speedup.
    """
    if not reports:
        raise ValueError("pick_best needs at least one report")
    return min(
        reports,
        key=lambda report: (
            report.serving_stats.max_ms,
            -report.speedup_vs_eager,
        ),
    )


__all__ = ["Experiment", "Mitigation", "SolutionOutcome", "pick_best"]
