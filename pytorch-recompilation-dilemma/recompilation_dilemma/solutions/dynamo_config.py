"""Option 3 -- configure Dynamo so the guards stop firing.

When Dynamo reports a guard failure it also reports how to avoid the
specialisation that produced it.  For the toy model the two hints are:

``use_duck_shape``
    Dynamo observed that two edge tensors had the same size and added the
    equality ``cross_edge_index.size()[1] == local_edge_index.size()[1]`` as a
    guard.  Setting ``torch.fx.experimental._config.use_duck_shape = False``
    stops it inferring relationships between unrelated tensors.
``0/1 specialisation``
    Dynamo guards ``size >= 2`` because Python ints of 0 and 1 are special.  Using
    ``mark_unbacked`` on the variable dimensions tells it the size is
    data-dependent, so the check becomes a runtime assertion instead of a
    recompilation trigger.

This is the most effective and the most brittle of the six.  It is effective
because it removes the guard failures rather than paying for them; it is brittle
because the guards it removes are an artefact of the sample *ordering* that
produced them.  The module therefore measures both orderings: the one the
configuration was tuned against, and the same catalog with two entries swapped,
which reintroduces recompilations -- including a stride mismatch that no
configuration hint anticipates.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from recompilation_dilemma.config import ModelConfig, RunConfig
from recompilation_dilemma.dataset import shuffled_pattern
from recompilation_dilemma.harness import (
    Experiment,
    compile_model,
    dynamo_settings,
    log_recompiles,
    reset_compiler,
    run_compiled,
)
from recompilation_dilemma.metrics import LatencyReport
from recompilation_dilemma.solutions.base import SolutionOutcome, pick_best

OPTION = 3
KEY = "dynamo-config"
NAME = "Control compilation behaviour with Dynamo configuration"


@contextmanager
def duck_shape_disabled() -> Iterator[None]:
    """Turn off duck-shape equality guards for the duration of the block."""
    import torch.fx.experimental._config as fx_config

    previous = fx_config.use_duck_shape
    fx_config.use_duck_shape = False
    try:
        yield
    finally:
        fx_config.use_duck_shape = previous


def mark_sample_unbacked(
    sample: dict[str, torch.Tensor], model_config: ModelConfig
) -> list[str]:
    """Mark each variable request dimension unbacked.

    Returns the dimension names that were marked, for the report.  Only the
    dimensions that actually vary between requests are marked: node rows, edge
    counts and context rows.  The feature dimension is a fixed model property and
    is deliberately left alone -- marking it unbacked would be lying to the
    compiler about the model rather than describing the data.
    """
    from torch._dynamo.decorators import mark_unbacked

    variable_dimensions: tuple[tuple[str, int], ...] = (
        ("x", 0),
        ("local_edge_index", 1),
        ("cross_edge_index", 1),
        ("local_edge_scale", 0),
        ("cross_edge_scale", 0),
        ("context", 0),
    )
    marked: list[str] = []
    for name, dimension in variable_dimensions:
        mark_unbacked(sample[name], dimension)
        marked.append(f"{name}[{dimension}]")
    return marked


def prepare_data(experiment: Experiment, model_config: ModelConfig) -> tuple[list, list[str]]:
    """Return a request list whose first entry is marked unbacked.

    The mark has to land on the tensor that is actually traced, so the first
    request is cloned rather than marked in place.  Marking the shared dataset
    would leak: the marks live on the tensor objects, so a later mitigation
    running against the same list would inherit guards it never asked for -- and
    would appear to need no recompilations at all.
    """
    first = {name: tensor.clone() for name, tensor in experiment.dataset[0].items()}
    marked = mark_sample_unbacked(first, model_config)
    return [first, *experiment.dataset[1:]], marked


def _configured_run(
    config: RunConfig,
    experiment: Experiment,
    *,
    label: str,
    notes: list[str],
    verbose: bool,
) -> LatencyReport:
    data, marked = prepare_data(experiment, config.model)
    reset_compiler(config)
    with dynamo_settings(cache_size_limit=config.cache_size_limit), duck_shape_disabled():
        model = compile_model(experiment.fresh_model(), config)
        with log_recompiles(verbose):
            report = run_compiled(
                model,
                data,
                experiment.device,
                config=config,
                label=label,
                mode=f"{config.backend}+configured",
                eager_baseline_ms=experiment.eager_ms,
                notes=[
                    *notes,
                    "use_duck_shape=False; marked unbacked: " + ", ".join(marked),
                ],
            )
    return report


def run(experiment: Experiment, *, verbose: bool | None = None) -> SolutionOutcome:
    config = experiment.config
    verbose = config.verbose_recompiles if verbose is None else verbose

    tuned = _configured_run(
        config,
        experiment,
        label="Option 3: configured Dynamo (tuned ordering)",
        notes=["catalog order the guards were tuned against"],
        verbose=verbose,
    )

    # Same configuration, same catalog, two entries swapped.  The eager baseline
    # is re-measured because the shuffled catalog produces a different dataset.
    shuffled = Experiment(
        config,
        samples=len(experiment.dataset),
        pattern=shuffled_pattern,
    )
    shuffled_baseline = shuffled.eager_ms
    shuffled_data, marked = prepare_data(shuffled, config.model)
    reset_compiler(config)
    with dynamo_settings(cache_size_limit=config.cache_size_limit), duck_shape_disabled():
        model = compile_model(shuffled.fresh_model(), config)
        with log_recompiles(verbose):
            brittle = run_compiled(
                model,
                shuffled_data,
                shuffled.device,
                config=config,
                label="Option 3: same config, swapped catalog order",
                mode=f"{config.backend}+configured",
                eager_baseline_ms=shuffled_baseline,
                notes=[
                    "catalog entries 0 and 3 swapped -- nothing else changed",
                    "use_duck_shape=False; marked unbacked: " + ", ".join(marked),
                ],
            )

    best = pick_best([tuned, brittle], config.budget)
    outcome = SolutionOutcome(
        option=OPTION,
        key=KEY,
        name=NAME,
        report=best,
        tradeoff="removes the guard failures instead of paying for them",
        caveat=(
            "the configuration fits the sample ordering that produced the guard "
            "messages; a different order reintroduces recompilations, including "
            "ones no hint anticipates"
        ),
        variants=[tuned, brittle],
        notes=[
            f"tuned ordering   : {len(tuned.recompile_indices)} recompilation(s), "
            f"max {tuned.serving_stats.max_ms:.2f}ms",
            f"swapped ordering : {len(brittle.recompile_indices)} recompilation(s), "
            f"max {brittle.serving_stats.max_ms:.2f}ms",
        ],
    )
    return outcome
