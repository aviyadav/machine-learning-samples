"""Option 2 -- cap the number of recompilations.

``dynamo.config.cache_size_limit`` bounds how many compiled variants Dynamo
keeps for one code object.  Once the limit is reached, a request whose guards
cannot be satisfied is not recompiled: it simply runs eagerly.

That converts an unbounded tail into a bounded one -- at a price.  Requests that
miss are served at *eager* speed, so the setting controls a direct trade between
how many latency spikes are possible and how much of the dataset actually gets
the compiled path.  The limit therefore has to be tuned against the shape
distribution: on the article's data, a limit of 2 leaves only 9 requests on the
eager path, whereas a limit of 1 leaves 299.

``fullgraph=False`` is required: with the graph split into pieces, Dynamo can
fall back per-piece instead of failing the whole call.
"""

from __future__ import annotations

from recompilation_dilemma.config import RunConfig
from recompilation_dilemma.harness import (
    Experiment,
    compile_model,
    dynamo_settings,
    log_recompiles,
    reset_compiler,
    run_compiled,
)
from recompilation_dilemma.solutions.base import SolutionOutcome, pick_best

OPTION = 2
KEY = "cache-size-limit"
NAME = "Rely on the recompilation cache limit"

#: The limits the article tries.  Exposed so a caller can widen the sweep.
DEFAULT_LIMITS: tuple[int, ...] = (1, 2)


def _sweep(
    config: RunConfig,
    experiment: Experiment,
    limit: int,
    *,
    verbose: bool,
) -> object:
    reset_compiler(config)
    with dynamo_settings(cache_size_limit=limit):
        # fullgraph=False lets Dynamo fall back per graph piece when the cache
        # limit is exhausted instead of raising.
        model = compile_model(experiment.fresh_model(), config, fullgraph=False)
        with log_recompiles(verbose):
            return run_compiled(
                model,
                experiment.dataset,
                experiment.device,
                config=config,
                label=f"Option 2: cache_size_limit={limit}",
                mode=f"{config.backend}+fallback",
                eager_baseline_ms=experiment.eager_ms,
                notes=[f"fullgraph=False, cache_size_limit={limit}"],
            )


def run(
    experiment: Experiment,
    *,
    limits: tuple[int, ...] = DEFAULT_LIMITS,
    verbose: bool | None = None,
) -> SolutionOutcome:
    config = experiment.config
    verbose = config.verbose_recompiles if verbose is None else verbose

    variants = [_sweep(config, experiment, limit, verbose=verbose) for limit in limits]
    best = pick_best(variants, config.budget)

    notes = []
    for report in variants:
        notes.append(
            f"limit={report.label.rsplit('=', 1)[-1]}: "
            f"{len(report.recompile_indices)} recompilation(s), "
            f"{len(report.fallback_indices)} eager fallback(s), "
            f"max {report.serving_stats.max_ms:.2f}ms, "
            f"speedup {report.speedup_vs_eager:.2f}x"
        )

    return SolutionOutcome(
        option=OPTION,
        key=KEY,
        name=NAME,
        report=best,
        tradeoff="bounds the tail by giving up the compiled path for guard misses",
        caveat=(
            "the limit must be tuned to the shape distribution: too low and most "
            "requests never use the compiled graph at all"
        ),
        variants=variants,
        notes=notes,
    )
