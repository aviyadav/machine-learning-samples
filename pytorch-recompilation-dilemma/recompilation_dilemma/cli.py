"""Command line entry point.

Four commands:

``problem``
    Measure eager and naive ``torch.compile`` and report whether the dilemma
    reproduces in this environment.
``list``
    List the mitigations and their keys.
``solution``
    Run one mitigation by key or option number.
``all``
    Run the problem and every mitigation against one shared experiment, then
    print the comparison table.

Everything is judged by the same :class:`~recompilation_dilemma.config.LatencyBudgetSpec`,
so "PASS" always means the same thing: the tail bound held *and* the requests
were actually served by the compiled graph.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import replace

from recompilation_dilemma import problem
from recompilation_dilemma.config import LatencyBudgetSpec, Profile, RunConfig
from recompilation_dilemma.harness import Experiment, default_profile
from recompilation_dilemma.metrics import Verdict
from recompilation_dilemma.runtime import resolve_device
from recompilation_dilemma.solutions import MITIGATIONS, SolutionOutcome, resolve


def _build_config(args: argparse.Namespace) -> RunConfig:
    device = resolve_device(args.device)
    profile = default_profile(device, Profile(args.profile) if args.profile else None)
    return RunConfig.for_profile(
        profile,
        device=device.type,
        backend=args.backend,
        samples=args.samples,
        cache_size_limit=args.cache_size_limit,
    )


def _banner(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(str(row[column])) for row in [headers, *rows])
        for column in range(len(headers))
    ]
    lines = [
        "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(headers)).rstrip(),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        for row in rows
    )
    return "\n".join(lines)


def _verdict_cell(verdict: Verdict | None) -> str:
    if verdict is None:
        return "n/a"
    return verdict.symbol


def _outcome_rows(
    outcome: SolutionOutcome, budget: LatencyBudgetSpec
) -> list[list[str]]:
    report = outcome.report
    verdict = outcome.verdict(budget)
    if report is None:
        return [[str(outcome.option), outcome.name, "-", "-", "-", "-", "not run"]]
    return [
        [
            str(outcome.option),
            outcome.name,
            str(len(report.recompile_indices)),
            f"{report.compiled_ratio:.0%}",
            f"{report.serving_stats.max_ms:.0f}ms",
            f"{report.speedup_vs_eager:.2f}x",
            _verdict_cell(verdict),
        ]
    ]


#: Shared by the problem table and the end-of-run summary.
_SUMMARY_HEADERS = [
    "#",
    "mitigation",
    "recompiles",
    "compiled",
    "max step",
    "speedup",
    "verdict",
]


def _budget_line(budget: LatencyBudgetSpec) -> str:
    """Describe the budget in the same words the checks are implemented in."""
    parts = [f"no served request may exceed {budget.max_step_ms:.0f}ms"]
    parts.append(
        f"at least {budget.min_compiled_ratio:.0%} of served requests must use the "
        "compiled graph"
    )
    if budget.min_speedup is None:
        parts.append("no steady-state speedup is required")
    else:
        parts.append(f"steady-state speedup must be at least {budget.min_speedup:.2f}x")
    return "; ".join(parts)


def _print_outcome(outcome: SolutionOutcome, budget: LatencyBudgetSpec) -> None:
    _banner(outcome.headline())
    if outcome.report is not None:
        for line in outcome.report.summary_lines():
            print(f"  {line}")
    if outcome.variants:
        print()
        print("  variants:")
        for variant in outcome.variants:
            print(
                f"    - {variant.label}: {len(variant.recompile_indices)} recompile(s),"
                f" max {variant.serving_stats.max_ms:.0f}ms,"
                f" compiled {variant.compiled_ratio:.0%},"
                f" speedup {variant.speedup_vs_eager:.2f}x"
            )
    print()
    print(f"  tradeoff: {outcome.tradeoff}")
    print(f"  caveat  : {outcome.caveat}")
    for note in outcome.notes:
        print(f"  note    : {note}")
    verdict = outcome.verdict(budget)
    if verdict is not None:
        print()
        print(f"  verdict : {verdict.symbol}")
        for note in verdict.notes:
            print(f"            {textwrap.fill(note, 84, subsequent_indent=' ' * 12)}")


def command_problem(args: argparse.Namespace, config: RunConfig) -> int:
    experiment = Experiment(config, samples=args.samples)
    _banner("Environment")
    for line in experiment.describe():
        print(f"  {line}")

    dilemma = problem.run(experiment, verbose=args.verbose)
    _banner("The problem")
    for line in dilemma.describe():
        print(f"  {line}")

    _banner("Budget check")
    print(
        _table(
            ["configuration", "compiled", "max step", "speedup", "verdict"],
            [
                [
                    "eager",
                    f"{dilemma.eager.compiled_ratio:.0%}",
                    f"{dilemma.eager.serving_stats.max_ms:.0f}ms",
                    f"{dilemma.eager.speedup_vs_eager:.2f}x",
                    dilemma.eager_verdict.symbol,
                ],
                [
                    "torch.compile (as-is)",
                    f"{dilemma.compiled.compiled_ratio:.0%}",
                    f"{dilemma.compiled.serving_stats.max_ms:.0f}ms",
                    f"{dilemma.compiled.speedup_vs_eager:.2f}x",
                    dilemma.compiled_verdict.symbol,
                ],
                [
                    "budget",
                    f">={config.budget.min_compiled_ratio:.0%}",
                    f"{config.budget.max_step_ms:.0f}ms",
                    (
                        f">={config.budget.min_speedup:.2f}x"
                        if config.budget.min_speedup is not None
                        else "not required"
                    ),
                    "",
                ],
            ],
        )
    )
    print()
    if dilemma.demonstrated is True:
        print("The dilemma reproduces here: neither configuration meets the budget.")
    elif dilemma.demonstrated is None:
        print(
            "The dilemma is not evaluable here: a speedup is required but this "
            "backend generates no optimised code, so neither configuration is judged."
        )
    else:
        print("The dilemma did not reproduce here; see the notes above.")
    return 0


def run_mitigation(entry, experiment, *, verbose: bool) -> SolutionOutcome:
    """Run one mitigation, turning a raised exception into a reported outcome.

    A matrix run is meant to survive one mitigation failing: the failure is
    reported where it happened rather than swallowed, but it does not take the
    other five measurements down with it.  ``SystemExit`` and ``KeyboardInterrupt``
    are deliberately not caught.
    """
    try:
        return entry.run(experiment, verbose=verbose)
    except Exception as error:  # noqa: BLE001 - reported, not hidden
        first_line = str(error).splitlines()[0] if str(error) else ""
        return SolutionOutcome(
            option=entry.option,
            key=entry.key,
            name=entry.name,
            report=None,
            tradeoff="",
            caveat="",
            unavailable=(
                f"raised {type(error).__name__}"
                + (f": {first_line}" if first_line else "")
            ),
        )


def command_solution(args: argparse.Namespace, config: RunConfig) -> int:
    entry = resolve(args.name)
    experiment = Experiment(config, samples=args.samples)
    outcome = run_mitigation(entry, experiment, verbose=args.verbose)
    _print_outcome(outcome, config.budget)
    return 0 if outcome.ran else 1


def command_all(args: argparse.Namespace, config: RunConfig) -> int:
    experiment = Experiment(config, samples=args.samples)

    _banner("Environment")
    for line in experiment.describe():
        print(f"  {line}")

    dilemma = problem.run(experiment, verbose=args.verbose)
    _banner("The problem")
    for line in dilemma.describe():
        print(f"  {line}")

    outcomes: list[SolutionOutcome] = []
    for entry in MITIGATIONS:
        outcome = run_mitigation(entry, experiment, verbose=args.verbose)
        outcomes.append(outcome)
        _print_outcome(outcome, config.budget)

    _banner("Summary")
    rows: list[list[str]] = [
        [
            "0",
            "eager (no compilation)",
            "-",
            f"{dilemma.eager.compiled_ratio:.0%}",
            f"{dilemma.eager.serving_stats.max_ms:.0f}ms",
            f"{dilemma.eager.speedup_vs_eager:.2f}x",
            dilemma.eager_verdict.symbol,
        ],
        [
            "0",
            "torch.compile (as-is)",
            str(len(dilemma.compiled.recompile_indices)),
            f"{dilemma.compiled.compiled_ratio:.0%}",
            f"{dilemma.compiled.serving_stats.max_ms:.0f}ms",
            f"{dilemma.compiled.speedup_vs_eager:.2f}x",
            dilemma.compiled_verdict.symbol,
        ],
    ]
    for outcome in outcomes:
        rows.extend(_outcome_rows(outcome, config.budget))

    print(_table(_SUMMARY_HEADERS, rows))
    print()
    print(f"  budget: {_budget_line(config.budget)}")
    if not config.optimizing_backend:
        print(
            f"  note: backend={config.backend!r} runs the traced graph without code "
            "generation, so the speedup column is not meaningful; the recompilation "
            "accounting still is."
        )
    passed = [
        outcome.name
        for outcome in outcomes
        if (verdict := outcome.verdict(config.budget)) is not None and verdict.ok
    ]
    if passed:
        print(f"  mitigations that met the budget: {', '.join(passed)}")
    else:
        print("  no mitigation met the full budget in this environment.")
    return 0


def command_list(args: argparse.Namespace) -> int:
    del args
    for entry in MITIGATIONS:
        print(f"{entry.option}  {entry.key:<20} {entry.name}")
    return 0


def add_shared_options(parser: argparse.ArgumentParser, *, suppress_defaults: bool) -> None:
    """Register the options every command shares.

    They are registered twice on purpose -- once on the root parser and once on
    each subparser -- so both ``... --profile smoke all`` and
    ``... all --profile smoke`` work.  The subparser copies use ``SUPPRESS`` as
    their default so that omitting them there does not clobber a value already
    parsed by the root parser.
    """

    def default(value: object) -> object:
        return argparse.SUPPRESS if suppress_defaults else value

    parser.add_argument(
        "--profile",
        choices=[profile.value for profile in Profile],
        default=default(None),
        help="article = the article's tensor sizes; smoke = laptop-sized (default: by device)",
    )
    parser.add_argument("--device", default=default("auto"), help="auto, cpu or cuda")
    parser.add_argument(
        "--backend",
        default=default("inductor"),
        help="torch.compile backend; 'eager' exercises the recompilation machinery "
        "without needing a C++ toolchain",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=default(None),
        help="override the dataset size",
    )
    parser.add_argument(
        "--cache-size-limit",
        type=int,
        default=default(64),
        help="torch._dynamo.config.cache_size_limit for the unmitigated run",
    )
    parser.add_argument(
        "--budget-max-step-ms",
        type=float,
        default=default(None),
        help="override the tail bound of the latency budget",
    )
    parser.add_argument(
        "--budget-min-compiled-ratio",
        type=float,
        default=default(None),
        help="override the minimum fraction of requests that must use the compiled graph",
    )
    parser.add_argument(
        "--budget-min-speedup",
        type=float,
        default=default(None),
        help="require a steady-state speedup over eager (off by default on CPU)",
    )
    parser.add_argument(
        "--quiet",
        dest="verbose",
        action="store_false",
        default=default(True),
        help="suppress Dynamo recompile logs",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recompilation_dilemma",
        description=(
            "Reproduce the PyTorch recompilation dilemma and measure six "
            "mitigations against one latency budget."
        ),
    )
    add_shared_options(parser, suppress_defaults=False)

    # Subparsers accept the same options so they can be written after the
    # command name.
    common = argparse.ArgumentParser(add_help=False)
    add_shared_options(common, suppress_defaults=True)

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("problem", parents=[common], help="measure the dilemma")
    sub.add_parser("list", parents=[common], help="list the mitigations")

    solution = sub.add_parser(
        "solution", parents=[common], help="run one mitigation"
    )
    solution.add_argument("name", help="mitigation key or option number")

    sub.add_parser(
        "all", parents=[common], help="run the problem and every mitigation"
    )
    return parser


def _apply_budget_overrides(config: RunConfig, args: argparse.Namespace) -> RunConfig:
    overrides = (
        args.budget_max_step_ms,
        args.budget_min_compiled_ratio,
        args.budget_min_speedup,
    )
    if all(value is None for value in overrides):
        return config

    budget = LatencyBudgetSpec(
        max_step_ms=(
            args.budget_max_step_ms
            if args.budget_max_step_ms is not None
            else config.budget.max_step_ms
        ),
        min_compiled_ratio=(
            args.budget_min_compiled_ratio
            if args.budget_min_compiled_ratio is not None
            else config.budget.min_compiled_ratio
        ),
        min_speedup=(
            args.budget_min_speedup
            if args.budget_min_speedup is not None
            else config.budget.min_speedup
        ),
    )
    # ``replace`` rather than a hand-written copy of every field: a new field on
    # RunConfig would otherwise be silently dropped here.
    return replace(config, budget=budget)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        return command_list(args)

    config = _apply_budget_overrides(_build_config(args), args)
    if args.command == "problem":
        return command_problem(args, config)
    if args.command == "solution":
        return command_solution(args, config)
    if args.command == "all":
        return command_all(args, config)
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
