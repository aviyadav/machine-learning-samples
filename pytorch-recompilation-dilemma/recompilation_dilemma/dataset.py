"""The synthetic graph workload, and the injection that makes it recompile.

A dataset of random graphs would mostly produce *one* shape family, because
Dynamo specialises on the first few sizes it sees and then duck-sizes the rest.
To get the ten distinct recompilations the article reports, ten specific request
indices are replaced by "catalog" topologies that violate each other's guards:
empty relations, empty context, empty graphs, and equal edge counts.

The catalog is what turns a benchmark into a reproduction: samples 4, 12, 24, ...
are the ones that will trigger a compile, and every mitigation is judged on how
it handles them.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

import torch

from recompilation_dilemma.config import DatasetConfig

#: Request indices at which a catalog topology is injected.  Ten entries, so a
#: naive compiled run recompiles ten times.
INJECT_ID: tuple[int, ...] = (0, 4, 12, 24, 40, 60, 84, 112, 144, 180)

#: Human-readable names for the catalog entries, used by the reports.
PATTERN_NAMES: tuple[str, ...] = (
    "equal edge counts",
    "local relation empty",
    "cross relation empty",
    "both relations empty",
    "context empty",
    "local relation + context empty",
    "cross relation + context empty",
    "nodes only",
    "node-empty graph, context present",
    "fully empty graph and context",
)


def _empty_edges(device: torch.device) -> torch.Tensor:
    """A (2, 0) long tensor with a deliberately unusual stride.

    Dynamo guards on strides as well as shapes.  Using an explicit strided
    construction -- rather than letting ``torch.empty`` pick a layout -- is what
    later produces the "stride mismatch" guard failure when the catalog order
    changes.
    """
    return torch.empty_strided((2, 0), (0, 1), dtype=torch.long, device=device)


def _make_edges(
    nodes: int,
    count: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Draw ``count`` random edges over ``nodes`` nodes."""
    if count == 0:
        return _empty_edges(device)
    return torch.stack(
        (
            torch.randint(nodes, (count,), generator=generator, device=device),
            torch.randint(nodes, (count,), generator=generator, device=device),
        )
    )


def apply_pattern(
    catalog_number: int | None,
    nodes: int,
    local_edges: int,
    cross_edges: int,
    context_rows: int,
) -> tuple[int, int, int, int]:
    """Override the four random counts with a catalog topology."""
    # Non-catalog request: keep four independent, positive random counts.
    if catalog_number is None:
        return nodes, local_edges, cross_edges, context_rows
    if catalog_number == 0:
        # Both relations have the same edge count.
        return nodes, local_edges, local_edges, context_rows
    if catalog_number == 1:
        # Local relation empty.
        return nodes, 0, cross_edges, context_rows
    if catalog_number == 2:
        # Cross relation empty.
        return nodes, local_edges, 0, context_rows
    if catalog_number == 3:
        # Both relations empty.
        return nodes, 0, 0, context_rows
    if catalog_number == 4:
        # Context empty.
        return nodes, local_edges, cross_edges, 0
    if catalog_number == 5:
        # Local relation and context empty.
        return nodes, 0, cross_edges, 0
    if catalog_number == 6:
        # Cross relation and context empty.
        return nodes, local_edges, 0, 0
    if catalog_number == 7:
        # Nodes only: both relations and context are empty.
        return nodes, 0, 0, 0
    if catalog_number == 8:
        # Node-empty graph with context still present.
        return 0, 0, 0, context_rows
    if catalog_number == 9:
        # Fully empty graph and context.
        return 0, 0, 0, 0
    raise ValueError(f"unknown catalog pattern: {catalog_number}")


def shuffled_pattern(catalog_number: int | None, *args: int) -> tuple[int, int, int, int]:
    """The same catalog with entries 0 and 3 swapped.

    Used by the Dynamo-configuration experiment to show that a mitigation tuned
    against one sample ordering does not survive a different one.
    """
    if catalog_number == 0:
        catalog_number = 3
    elif catalog_number == 3:
        catalog_number = 0
    return apply_pattern(catalog_number, *args)


def _make_sample(
    device: torch.device,
    nodes: int,
    local_edges: int,
    cross_edges: int,
    context_rows: int,
    tensor_rng: torch.Generator,
    feature_dim: int,
) -> dict[str, torch.Tensor]:
    """Materialise a graph from its four row/edge counts."""
    return {
        "x": torch.randn(nodes, feature_dim, generator=tensor_rng, device=device),
        "local_edge_index": _make_edges(
            nodes, local_edges, generator=tensor_rng, device=device
        ),
        "cross_edge_index": _make_edges(
            nodes, cross_edges, generator=tensor_rng, device=device
        ),
        "local_edge_scale": torch.randn(
            local_edges, 1, generator=tensor_rng, device=device
        ),
        "cross_edge_scale": torch.randn(
            cross_edges, 1, generator=tensor_rng, device=device
        ),
        "context": torch.randn(
            context_rows, feature_dim, generator=tensor_rng, device=device
        ),
    }


def _draw_counts(
    shape_rng: random.Random, config: DatasetConfig
) -> tuple[int, int, int, int]:
    return (
        shape_rng.randint(config.min_size, config.max_size),
        shape_rng.randint(config.min_size, config.max_size),
        shape_rng.randint(config.min_size, config.max_size),
        shape_rng.randint(config.min_size, config.max_size),
    )


def make_dataset(
    device: torch.device,
    config: DatasetConfig | None = None,
    *,
    feature_dim: int = 16,
    pattern: Callable[..., tuple[int, int, int, int]] = apply_pattern,
) -> list[dict[str, torch.Tensor]]:
    """Build the evaluated dataset, catalog injections included."""
    config = config or DatasetConfig()
    tensor_rng = torch.Generator(device=device).manual_seed(config.seed)
    shape_rng = random.Random(config.seed)

    catalog_by_request = {
        request_index: catalog_number
        for catalog_number, request_index in enumerate(INJECT_ID)
    }

    dataset: list[dict[str, torch.Tensor]] = []
    for request_index in range(config.samples):
        nodes, local_edges, cross_edges, context_rows = _draw_counts(shape_rng, config)

        # Scheduled catalog requests override the random positive counts.
        nodes, local_edges, cross_edges, context_rows = pattern(
            catalog_by_request.get(request_index),
            nodes,
            local_edges,
            cross_edges,
            context_rows,
        )

        dataset.append(
            _make_sample(
                device,
                nodes=nodes,
                local_edges=local_edges,
                cross_edges=cross_edges,
                context_rows=context_rows,
                tensor_rng=tensor_rng,
                feature_dim=feature_dim,
            )
        )
    return dataset


def warmup_dataset(
    device: torch.device,
    config: DatasetConfig | None = None,
    *,
    feature_dim: int = 16,
    pattern: Callable[..., tuple[int, int, int, int]] = apply_pattern,
    seed: int = 1_017,
) -> list[dict[str, torch.Tensor]]:
    """One deterministic representative for every catalog pattern.

    This is the warm-up set for the "seed the compiler" mitigation: run the
    model over these before serving, and every topology the catalog can produce
    is already in the Dynamo cache.
    """
    config = config or DatasetConfig()
    tensor_rng = torch.Generator(device=device).manual_seed(seed)
    shape_rng = random.Random(seed)

    warmup: list[dict[str, torch.Tensor]] = []
    for catalog_number in range(len(INJECT_ID)):
        nodes, local_edges, cross_edges, context_rows = _draw_counts(shape_rng, config)
        nodes, local_edges, cross_edges, context_rows = pattern(
            catalog_number, nodes, local_edges, cross_edges, context_rows
        )
        warmup.append(
            _make_sample(
                device,
                nodes=nodes,
                local_edges=local_edges,
                cross_edges=cross_edges,
                context_rows=context_rows,
                tensor_rng=tensor_rng,
                feature_dim=feature_dim,
            )
        )
    return warmup


def catalog_indices(samples: Sequence[int]) -> list[int]:
    """Which of ``samples`` request indices carry a catalog injection."""
    return [index for index in INJECT_ID if index < samples]
