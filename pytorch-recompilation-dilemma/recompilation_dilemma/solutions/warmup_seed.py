"""Option 4 -- seed the compiler with a warm-up pass.

Instead of trying to stop recompilations, this mitigation schedules them.  The
service compiles every topology it expects to see *before* it starts serving, so
by the time real traffic arrives the graphs are already in the Dynamo cache and
the serving loop never pays for a compilation.

The measurement therefore has two parts, and both matter:

* the warm-up cost itself -- it is paid once, at start-up, and is the price of
  the guarantee;
* the serving loop, which should show zero recompilations.

The assumption it rests on is total knowledge of the input topologies.  A missed
topology does not degrade gracefully: it recompiles mid-flight exactly as in the
unmitigated case, which is why the note about coverage matters more than the
speedup.
"""

from __future__ import annotations

import torch

from recompilation_dilemma.dataset import INJECT_ID, PATTERN_NAMES, warmup_dataset
from recompilation_dilemma.harness import (
    Experiment,
    compile_model,
    dynamo_settings,
    log_recompiles,
    reset_compiler,
    run_compiled,
)
from recompilation_dilemma.runtime import StepTimer
from recompilation_dilemma.solutions.base import SolutionOutcome

OPTION = 4
KEY = "warmup-seed"
NAME = "Seed the graph compiler during a warm-up phase"


def run(experiment: Experiment, *, verbose: bool | None = None) -> SolutionOutcome:
    config = experiment.config
    verbose = config.verbose_recompiles if verbose is None else verbose

    reset_compiler(config)
    with dynamo_settings(cache_size_limit=config.cache_size_limit):
        model = compile_model(experiment.fresh_model(), config)

        samples = warmup_dataset(
            experiment.device,
            config.dataset,
            feature_dim=config.model.feature_dim,
            pattern=experiment.pattern,
        )

        with StepTimer(experiment.device) as warmup_timer, torch.inference_mode():
            for sample in samples:
                model(**sample)
        warmup_ms = warmup_timer.elapsed_ms

        with log_recompiles(verbose):
            report = run_compiled(
                model,
                experiment.dataset,
                experiment.device,
                config=config,
                label="Option 4: warm-up seeded compiler",
                mode=config.backend,
                eager_baseline_ms=experiment.eager_ms,
                notes=[
                    f"warm-up compiled {len(samples)} topology/topologies "
                    f"in {warmup_ms:.0f}ms ({warmup_ms / len(samples):.0f}ms each)"
                ],
            )

    covered = len(samples) >= len(INJECT_ID)
    outcome = SolutionOutcome(
        option=OPTION,
        key=KEY,
        name=NAME,
        report=report,
        tradeoff="moves every expected compilation from the serving loop into start-up",
        caveat=(
            "requires complete knowledge of the input topologies up front; any "
            "topology the warm-up misses recompiles mid-flight"
            if covered
            else f"only {len(samples)} of {len(INJECT_ID)} catalog topologies were warmed"
        ),
        notes=[
            "warmed topologies: " + "; ".join(PATTERN_NAMES[: len(samples)]),
            f"warm-up cost: {warmup_ms:.0f}ms before the first request is served",
        ],
    )
    if report.recompile_indices and covered:
        # Worth stating precisely: the catalog was covered in full, so this is not
        # a coverage failure.  Dynamo simply did not serve the first request from
        # a warm-up graph, which is the part of the technique the article's
        # caveat is about.
        outcome.notes.append(
            f"all {len(INJECT_ID)} catalog topologies were warmed, yet serving "
            f"still compiled at request(s) {report.recompile_indices}: the warm-up "
            "graphs were not reused for those requests"
        )
    elif report.recompile_indices:
        outcome.notes.append(
            "warm-up did not cover everything: "
            f"recompilations at {report.recompile_indices}"
        )
    return outcome
