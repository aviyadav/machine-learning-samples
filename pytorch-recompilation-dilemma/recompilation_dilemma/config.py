"""Experiment configuration.

The article's numbers come from an Amazon EC2 g6e.2xlarge (NVIDIA L40S, PyTorch
2.13) running an AWS Deep Learning AMI.  Two profiles drive the same code:

``ARTICLE``
    The exact model and dataset sizes used in the article.  Needs a GPU to
    finish in reasonable time.
``SMOKE``
    A scaled-down version of the same workload, calibrated so the trade-off is
    visible on one CPU core.  Recompilation is triggered by Dynamo's guards on
    input shapes, so the interference pattern survives the scaling even though
    the absolute numbers do not.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Profile(str, Enum):
    """Pre-baked model and dataset sizes."""

    ARTICLE = "article"
    SMOKE = "smoke"


@dataclass(frozen=True)
class ModelConfig:
    """Dimensions of the toy message-passing GNN."""

    feature_dim: int = 16
    hidden_dim: int = 768
    message_passing_layers: int = 16

    def __post_init__(self) -> None:
        for name in ("feature_dim", "hidden_dim", "message_passing_layers"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")


@dataclass(frozen=True)
class DatasetConfig:
    """Shape distribution of the synthetic graph workload."""

    samples: int = 300
    seed: int = 17
    min_size: int = 1_000
    max_size: int = 4_000
    """Exclusive upper bound on the random row counts drawn per sample."""

    def __post_init__(self) -> None:
        if self.samples < 1:
            raise ValueError("samples must be positive")
        if not 0 < self.min_size < self.max_size:
            raise ValueError("need 0 < min_size < max_size")


@dataclass(frozen=True)
class LatencyBudgetSpec:
    """The service-level objective the serving loop has to meet.

    Three checks.  The first two are the ones that pull in opposite directions,
    which is the whole point:

    ``max_step_ms``
        No single request may be slower than this.  This is the hard real-time
        bound that a recompilation spike blows straight through.
    ``min_compiled_ratio``
        At least this fraction of requests must be served by the compiled graph.
        Without it, "stop compiling" would pass the tail bound trivially and the
        dilemma would not exist; it is also exactly what a recompilation cache
        limit trades away when it is set too low.
    ``min_speedup``
        Optionally, the steady-state mean must be at least this many times faster
        than eager.  This is the article's measurement, and it is left as ``None``
        for the CPU profile: whether Inductor beats eager at all is a property of
        the hardware, and on a laptop CPU the honest answer is that it roughly
        ties.  The tail and coverage checks are hardware-independent, so the
        dilemma still has teeth without it.
    """

    max_step_ms: float = 50.0
    min_compiled_ratio: float = 0.90
    min_speedup: float | None = None

    def __post_init__(self) -> None:
        if self.max_step_ms <= 0:
            raise ValueError("max_step_ms must be positive")
        if not 0.0 <= self.min_compiled_ratio <= 1.0:
            raise ValueError("min_compiled_ratio must be in [0, 1]")
        if self.min_speedup is not None and self.min_speedup < 1.0:
            raise ValueError("min_speedup must be >= 1.0 when it is set")


@dataclass(frozen=True)
class RunConfig:
    """Everything the experiments need to know about their environment."""

    device: str = "cpu"
    backend: str = "inductor"
    """Dynamo backend.  ``inductor`` is the article's choice; ``eager`` runs the
    traced graph without code generation so the recompilation machinery can be
    exercised on a machine with no C++ toolchain."""

    model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = DatasetConfig()
    budget: LatencyBudgetSpec = LatencyBudgetSpec()
    cache_size_limit: int = 64
    """Article raises this from the default of 8 to 64."""

    warmup_samples: int = 10
    cache_dir: str = ".rd-cache"
    """Where Inductor/Triton artifacts and the AOT package are written.

    A relative path is resolved against the repository root, not the working
    directory, so a run started from anywhere uses -- and cleans -- one cache.
    The name is kept deliberately short because Inductor nests several hashed
    directories beneath it and some filesystems cap path length; on Windows the
    AOT packaging step otherwise fails with a "path not found" that has nothing
    to do with your code.
    """
    verbose_recompiles: bool = True

    @classmethod
    def for_profile(
        cls,
        profile: Profile,
        *,
        device: str = "cpu",
        backend: str = "inductor",
        samples: int | None = None,
        cache_size_limit: int = 64,
    ) -> RunConfig:
        """Build a config for one of the two pre-baked profiles."""
        if profile is Profile.ARTICLE:
            model = ModelConfig()
            dataset = DatasetConfig(samples=samples or 300)
            budget = LatencyBudgetSpec(max_step_ms=50.0, min_speedup=1.10)
        elif profile is Profile.SMOKE:
            # A scaled-down version of the same workload.  Both the model *and*
            # the row counts are reduced, because on a CPU either extreme breaks
            # the measurement: with article-sized rows a single step takes long
            # enough that eager is no longer launch-bound, and with very few rows
            # the compiled path wins for the wrong reason.  These numbers were
            # calibrated so that the trade-off the article describes is visible
            # on one core -- roughly 1.3x on the warm path, with multi-second
            # spikes on the catalog topologies.
            model = ModelConfig(feature_dim=16, hidden_dim=64, message_passing_layers=2)
            dataset = DatasetConfig(samples=samples or 120, min_size=128, max_size=1_024)
            # The same 50ms bound the article's profile uses, which is what makes
            # the budget meaningful: a request must not take ~50x longer than the
            # eager mean because the compiler decided to work.  ``min_speedup`` is
            # deliberately left unset: Inductor on a laptop CPU roughly ties with
            # eager for a model this size, so demanding a speedup here would fail
            # every configuration and measure the hardware rather than the
            # mitigation.  The tail bound and the compiled-path coverage check
            # carry the dilemma instead, and neither depends on the hardware.
            budget = LatencyBudgetSpec(
                max_step_ms=50.0, min_compiled_ratio=0.90, min_speedup=None
            )
        else:  # pragma: no cover - guarded by the Profile enum
            raise ValueError(f"unknown profile: {profile!r}")

        return cls(
            device=device,
            backend=backend,
            model=model,
            dataset=dataset,
            budget=budget,
            cache_size_limit=cache_size_limit,
        )

    @property
    def optimizing_backend(self) -> bool:
        """Whether the backend is expected to produce a real speedup.

        ``eager`` keeps Dynamo's guards and recompilation accounting but skips
        code generation, so a speedup measured with it is not meaningful.  The
        report surfaces this instead of pretending otherwise.
        """
        return self.backend not in {"eager"}
