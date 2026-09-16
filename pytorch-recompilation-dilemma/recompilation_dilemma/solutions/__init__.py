"""The six mitigations, in the order the article presents them.

Each module exposes ``run(experiment) -> SolutionOutcome``.  The registry exists
so the CLI can run one by name or the whole set, and so the set is enumerable
rather than hard-coded in the command line.
"""

from __future__ import annotations

from dataclasses import dataclass

from recompilation_dilemma.solutions import (
    aot_export,
    async_background,
    cache_size_limit,
    dynamo_config,
    preload_cache,
    warmup_seed,
)
from recompilation_dilemma.solutions.base import Mitigation, SolutionOutcome


@dataclass(frozen=True)
class Entry:
    key: str
    option: int
    name: str
    run: Mitigation


MITIGATIONS: tuple[Entry, ...] = (
    Entry(preload_cache.KEY, preload_cache.OPTION, preload_cache.NAME, preload_cache.run),
    Entry(
        cache_size_limit.KEY,
        cache_size_limit.OPTION,
        cache_size_limit.NAME,
        cache_size_limit.run,
    ),
    Entry(dynamo_config.KEY, dynamo_config.OPTION, dynamo_config.NAME, dynamo_config.run),
    Entry(warmup_seed.KEY, warmup_seed.OPTION, warmup_seed.NAME, warmup_seed.run),
    Entry(
        async_background.KEY,
        async_background.OPTION,
        async_background.NAME,
        async_background.run,
    ),
    Entry(aot_export.KEY, aot_export.OPTION, aot_export.NAME, aot_export.run),
)

BY_KEY: dict[str, Entry] = {entry.key: entry for entry in MITIGATIONS}


def resolve(name: str) -> Entry:
    """Look a mitigation up by key or by option number."""
    if name in BY_KEY:
        return BY_KEY[name]
    if name.isdigit():
        number = int(name)
        for entry in MITIGATIONS:
            if entry.option == number:
                return entry
    known = ", ".join(entry.key for entry in MITIGATIONS)
    raise KeyError(f"unknown mitigation {name!r}; expected one of: {known}")


__all__ = ["BY_KEY", "MITIGATIONS", "Entry", "SolutionOutcome", "resolve"]
