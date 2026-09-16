"""Option 1 -- preload an existing compilation cache.

``torch.compiler.save_cache_artifacts`` / ``load_cache_artifacts`` move the
compiler's *code* caches (Inductor kernels, AOTAutograd graphs) between runs.  A
new process can therefore start with warmed artifacts instead of compiling from
scratch.

What it does not move is the part that matters most: the Dynamo runtime state
and its guards.  Those live in process memory and are rebuilt from nothing, so
the guards still fire and the recompilations still happen -- they are merely
cheaper, because the expensive backend work is a cache hit.

That distinction is the whole content of this mitigation, and it is why the
report separates "recompilations still triggered" from "recompilation cost".
"""

from __future__ import annotations

import torch

from recompilation_dilemma.config import RunConfig
from recompilation_dilemma.harness import (
    Experiment,
    compile_model,
    dynamo_settings,
    log_recompiles,
    reset_compiler,
    run_compiled,
)
from recompilation_dilemma.solutions.base import SolutionOutcome

OPTION = 1
KEY = "preload-cache"
NAME = "Preload an existing compilation cache"


def _previous_session(config: RunConfig, experiment: Experiment) -> tuple[bytes, int] | None:
    """Stand in for an earlier invocation of the same service.

    Runs a full pass so the catalog topologies are compiled, then serialises the
    resulting artifacts.  ``None`` means the backend produced nothing to save --
    which is itself worth reporting rather than papering over.
    """
    reset_compiler(config)
    model = compile_model(experiment.fresh_model(), config)
    # Raising the cache limit matters here as much as anywhere: the previous
    # session is expected to compile every catalog topology, and with the default
    # limit of 8 a fullgraph compile gives up instead of recompiling.
    with dynamo_settings(cache_size_limit=config.cache_size_limit), log_recompiles(False):
        for sample in experiment.dataset:
            with torch.inference_mode():
                model(**sample)

    saved = torch.compiler.save_cache_artifacts()
    if saved is None:
        return None
    artifacts, cache_info = saved
    if not artifacts:
        return None
    total = sum(len(entries) for entries in cache_info.artifacts.values())
    return artifacts, total


def run(experiment: Experiment, *, verbose: bool | None = None) -> SolutionOutcome:
    config = experiment.config
    verbose = config.verbose_recompiles if verbose is None else verbose

    session = _previous_session(config, experiment)
    if session is None:
        return SolutionOutcome(
            option=OPTION,
            key=KEY,
            name=NAME,
            report=None,
            tradeoff="replaces backend code generation with a cache lookup",
            caveat="only helps topologies a previous run already compiled",
            unavailable=(
                f"backend={config.backend!r} produced no cache artifacts to save "
                "-- load_cache_artifacts only carries Inductor/AOTAutograd code"
            ),
        )

    artifacts, artifact_count = session
    artifact_mb = len(artifacts) / (1024 * 1024)

    # A fresh process in every way that matters: Dynamo state, in-memory cache
    # and the on-disk cache directories are all cleared before loading.
    reset_compiler(config)
    load_info = torch.compiler.load_cache_artifacts(artifacts)
    loaded_count = (
        sum(len(entries) for entries in load_info.artifacts.values())
        if load_info is not None
        else 0
    )

    model = compile_model(experiment.fresh_model(), config)
    with dynamo_settings(
        cache_size_limit=config.cache_size_limit
    ), log_recompiles(verbose):
        report = run_compiled(
            model,
            experiment.dataset,
            experiment.device,
            config=config,
            label="Option 1: preloaded cache",
            mode=config.backend,
            eager_baseline_ms=experiment.eager_ms,
            notes=[
                f"loaded {loaded_count} cache artifact(s) ({artifact_mb:.1f} MiB)",
                "Dynamo guards are NOT part of the artifacts, so recompilations "
                "still trigger -- only their cost drops",
            ],
        )

    outcome = SolutionOutcome(
        option=OPTION,
        key=KEY,
        name=NAME,
        report=report,
        tradeoff="replaces backend code generation with a cache lookup",
        caveat="only helps topologies a previous run already compiled",
        notes=[
            f"previous session saved {artifact_count} artifact(s), {artifact_mb:.1f} MiB",
        ],
    )
    spikes = report.spike_stats
    if spikes is not None:
        outcome.notes.append(
            f"recompilations still fired {len(report.recompile_indices)} time(s); "
            f"worst recompilation took {spikes.max_ms:.0f}ms"
        )
    return outcome
