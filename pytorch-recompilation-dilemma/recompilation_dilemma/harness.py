"""Shared experiment machinery: compiler state, recompilation accounting, loops.

Everything the six mitigations have in common lives here, so that a mitigation
module only has to express what makes it *different*.

Two pieces of accounting deserve a word:

``unique_graph_count``
    Dynamo publishes ``stats/unique_graphs`` in ``torch._dynamo.utils.counters``.
    Sampling it before and after a request tells us whether that request caused a
    compilation.  Comparing against the previous requests separates the *initial*
    compilation from genuine *re*compilations.
``Model.eager_passes``
    Mirrors the article's counter.  The model only increments it when
    ``torch.compiler.is_compiling()`` is false, so under a hybrid scheme it counts
    exactly the requests that fell back to eager execution after a guard miss.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import shutil
import sys
import sysconfig
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import torch
import torch._dynamo as dynamo
import torch._dynamo.utils as dynamo_utils
import torch.fx.experimental._config as fx_config
import torch.nn as nn

from recompilation_dilemma.config import Profile, RunConfig
from recompilation_dilemma.dataset import apply_pattern
from recompilation_dilemma.metrics import LatencyReport, Step
from recompilation_dilemma.model import build_model
from recompilation_dilemma.runtime import StepTimer, resolve_device

#: A catalog-override function, as accepted by ``dataset.make_dataset``.
Pattern = Callable[..., tuple[int, int, int, int]]

#: Dynamo's configuration as it stood when this module was imported.
#: ``reset_compiler`` restores it between measurements, because several of the
#: mitigations are implemented precisely by changing these values -- and a
#: setting left behind by one measurement silently changes the next one.
_DYNAMO_DEFAULTS: dict[str, object] = dynamo.config.get_config_copy()
_DUCK_SHAPE_DEFAULT: bool = bool(fx_config.use_duck_shape)


# --------------------------------------------------------------------------
# Compiler state
# --------------------------------------------------------------------------


def unique_graph_count() -> int:
    """Number of distinct graphs Dynamo has compiled in this process."""
    return dynamo_utils.counters.get("stats", {}).get("unique_graphs", 0)


def cache_root(config: RunConfig) -> Path:
    """Absolute cache directory for this run.

    A relative ``cache_dir`` is resolved against the working directory, the same
    way every other tool treats a relative path.  Resolving it against the
    package directory instead would break a pip-installed copy, where that
    directory is site-packages and may be read-only.  :meth:`Experiment.describe`
    prints the resolved location, so it is never a guess which cache a run used.
    """
    return Path(config.cache_dir).expanduser().resolve()


def cache_paths(config: RunConfig) -> tuple[Path, Path]:
    """Inductor and Triton cache directories for this project.

    The article writes to ``/tmp/.cache/compile``; a run-local path is used here
    so the cache survives between runs and never collides with another
    workload's cache, or with a ``TORCHINDUCTOR_CACHE_DIR`` a user may have set
    for unrelated work.
    """
    root = cache_root(config)
    return root / "inductor", root / "triton"


def reset_compiler(config: RunConfig) -> None:
    """Return the process to a cold-compiler state.

    Clears the in-memory Dynamo cache and counters *and* the on-disk code
    caches, so a run never inherits compilation work from an earlier run.
    """
    # 1) In-memory Dynamo compilation cache and counters.
    dynamo.reset()
    dynamo_utils.counters.clear()

    # 2) Compiler configuration, back to the state this process started in.
    # Restoring a knob that this PyTorch version has deprecated is harmless, and
    # the deprecation warning for each of them would only be noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for key, value in _DYNAMO_DEFAULTS.items():
            try:
                setattr(dynamo.config, key, value)
            except (AttributeError, TypeError):  # pragma: no cover - read-only knobs
                continue
    fx_config.use_duck_shape = _DUCK_SHAPE_DEFAULT

    # 3) Deterministic on-disk cache paths, emptied.
    inductor_cache, triton_cache = cache_paths(config)
    shutil.rmtree(cache_root(config), ignore_errors=True)
    inductor_cache.mkdir(parents=True, exist_ok=True)
    triton_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)

    # 4) Generated Python modules cached in this process.  Only present when the
    # Inductor backend ships with this build, so its absence is not an error.
    try:
        from torch._inductor.codecache import PyCodeCache
    except ImportError:  # pragma: no cover - Inductor-less PyTorch build
        return
    PyCodeCache.cache_clear()


# --------------------------------------------------------------------------
# Native build diagnostics
# --------------------------------------------------------------------------


#: Marks a build hook this module has already wrapped, so importing it twice is safe.
_SHIM_MARKER = "_rd_python_header_first"


def _python_include_dir() -> Path | None:
    """Where this interpreter keeps ``Python.h`` -- the path Triton adds too."""
    include = Path(sysconfig.get_paths()["include"])
    return include if (include / "Python.h").is_file() else None


def silence_python_header_warnings() -> bool:
    """Put CPython's headers first in Triton's C builds, where they belong.

    Triton compiles a small C shim (``triton/backends/nvidia/driver.c``) the first
    time a kernel is launched, and that file includes ``<cuda.h>`` *before*
    ``Python.h``.  ``pyconfig.h`` therefore defines ``_GNU_SOURCE`` and
    ``_POSIX_C_SOURCE 200809L`` only once glibc has already picked its own defaults
    from ``<features.h>`` -- and since glibc 2.42 that default is
    ``_POSIX_C_SOURCE 202405L`` -- so gcc reports the second definition as a
    redefinition.  Triton does not capture its child compiler's stderr, so the
    warning lands in the middle of this project's output.

    No flag removes that diagnostic (``-Wno-macro-redefined`` is a clang option
    that gcc ignores), and defining the macro from the command line is not the
    answer either: ``driver.c`` calls ``alloca``, and without ``__USE_MISC`` -- which
    glibc only sets when the glibc feature macros are chosen before any header is
    read -- it is not declared.

    So the order is restored instead.  CPython's headers are meant to be included
    first, and ``Python.h`` is include-guarded, which makes force-including it a
    no-op for Triton's own ``#include <Python.h>``.  ``pyconfig.h`` then defines its
    macros before any glibc header is read, and when ``<features.h>`` does run it
    ``#undef``s them before defining its own, so no redefinition is reported
    anywhere and nothing is suppressed.  The declarations glibc exposes do not
    depend on which of the two definitions won -- they follow the ``__USE_*`` flags,
    which are the same either way -- so the compiled object is unchanged.

    Returns whether the shim was installed.  It does nothing off Linux, on a
    PyTorch without Triton, on a Triton whose build hook has another shape, or when
    this interpreter's ``Python.h`` cannot be found -- in which case the warning
    comes back rather than the build breaking.
    """
    if not sys.platform.startswith("linux"):
        return False
    python_include = _python_include_dir()
    if python_include is None:
        return False
    try:
        from triton.runtime import build as triton_build
    except ImportError:  # pragma: no cover - a PyTorch build without Triton
        return False

    original = getattr(triton_build, "_build", None)
    if original is None or getattr(original, _SHIM_MARKER, False):
        return False
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):  # pragma: no cover - not a Python callable
        return False
    if not {"ccflags", "language"} <= set(signature.parameters):
        return False

    def _build_python_header_first(*args: object, **kwargs: object) -> str:
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError:  # pragma: no cover - a call site this version never makes
            return original(*args, **kwargs)
        # Only C sources: the clash needs a glibc header ahead of Python.h, which is
        # what Triton's C shims have and its C++ ones do not.
        if bound.arguments.get("language", "c") != "c":
            return original(*args, **kwargs)
        bound.arguments["ccflags"] = [
            *(bound.arguments.get("ccflags") or []),
            # ``_DEFAULT_SOURCE`` keeps glibc's ``__USE_MISC`` on for the case where
            # this interpreter's ``pyconfig.h`` does not define ``_GNU_SOURCE``.
            "-D_DEFAULT_SOURCE",
            "-include",
            str(python_include / "Python.h"),
        ]
        return original(*bound.args, **bound.kwargs)

    setattr(_build_python_header_first, _SHIM_MARKER, True)
    triton_build._build = _build_python_header_first
    return True


@contextlib.contextmanager
def dynamo_settings(**overrides: object) -> Iterator[None]:
    """Temporarily override ``torch._dynamo.config`` values."""
    missing = [key for key in overrides if not hasattr(dynamo.config, key)]
    if missing:
        raise AttributeError(f"torch._dynamo.config has no {missing}")
    saved = {key: getattr(dynamo.config, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(dynamo.config, key, value)
        yield
    finally:
        for key, value in saved.items():
            setattr(dynamo.config, key, value)


@contextlib.contextmanager
def log_recompiles(enabled: bool) -> Iterator[None]:
    """Print PyTorch's own recompilation diagnostics to the console."""
    if not enabled:
        yield
        return
    torch._logging.set_logs(recompiles=True)
    try:
        yield
    finally:
        torch._logging.set_logs(recompiles=False)


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def measure_eager(model: nn.Module, data: list, device: torch.device) -> list[float]:
    """Per-step wall-clock latency of the eager baseline.

    This is the number every mitigation is compared against.  It is the *whole*
    dataset, first request included, because ``LatencyReport`` indexes into it by
    sample index.
    """
    timings: list[float] = []
    for sample in data:
        with StepTimer(device) as timer, torch.inference_mode():
            model(**sample)
        timings.append(timer.elapsed_ms)
    return timings


def compile_model(
    model: nn.Module,
    config: RunConfig,
    *,
    fullgraph: bool = True,
) -> nn.Module:
    """Wrap ``model`` with ``torch.compile`` using the project's settings."""
    return torch.compile(
        model,
        backend=config.backend,
        dynamic=True,
        fullgraph=fullgraph,
    )


def run_compiled(
    model: nn.Module,
    data: list,
    device: torch.device,
    *,
    config: RunConfig,
    label: str = "torch.compile",
    mode: str = "jit",
    eager_baseline_ms: list[float] | None = None,
    notes: list[str] | None = None,
) -> LatencyReport:
    """Serve ``data`` through a compiled model and record every step.

    This is the article's ``run_experiment``.  A step is marked ``new_graph``
    when it caused a compilation, and ``compiled=False`` when the model object
    fell back to eager execution for that request (which happens when the
    recompilation cache limit is exhausted).
    """
    model.reset_eager_passes()
    steps: list[Step] = []

    # Whether the compiler is cold.  If it is not -- because a previous session,
    # a preloaded cache or a warm-up phase already ran -- there is no "initial
    # compilation" left to excuse, so request 0 is served under load and every
    # new graph during serving is a genuine recompile.  The article's listing
    # assumed a cold start and granted the first new graph a free pass; that
    # assumption silently hides the very spike the warm-up is meant to remove.
    cold_start = unique_graph_count() == 0
    initial_compile_seen = not cold_start

    for index, sample in enumerate(data):
        before_graphs = unique_graph_count()
        before_eager = model.eager_passes

        with StepTimer(device) as timer, torch.inference_mode():
            model(**sample)

        new_graph = unique_graph_count() > before_graphs
        fell_back = model.eager_passes > before_eager

        # A new graph on the very first served request of a cold run is the
        # initial compilation, not a recompile.  Everything else is a recompile.
        is_recompile = new_graph and initial_compile_seen
        if new_graph:
            initial_compile_seen = True

        steps.append(
            Step(
                index=index,
                ms=timer.elapsed_ms,
                compiled=not fell_back,
                new_graph=is_recompile,
            )
        )

    report = LatencyReport(
        label=label,
        mode=mode,
        steps=steps,
        cold_start=cold_start,
        eager_baseline_ms=eager_baseline_ms,
        optimizing_backend=config.optimizing_backend,
        notes=list(notes or []),
    )
    if model.eager_passes:
        report.notes.append(
            f"{model.eager_passes} request(s) ran eagerly after a guard miss"
        )
    return report


# --------------------------------------------------------------------------
# Experiment context
# --------------------------------------------------------------------------


class Experiment:
    """Device, config and eager baseline, shared by every mitigation.

    Building this once per CLI invocation is what makes the six mitigations
    comparable: they all see the same weights, the same dataset, and the same
    eager reference numbers.
    """

    def __init__(
        self,
        config: RunConfig,
        *,
        samples: int | None = None,
        device: str | None = None,
        eager_seed: int = 0,
        pattern: Pattern | None = None,
    ) -> None:
        self.config = config
        self.device = resolve_device(device or config.device)
        self.model_config = config.model
        self.dataset_config = config.dataset
        self.pattern = pattern or apply_pattern
        self._eager_seed = eager_seed

        from recompilation_dilemma.dataset import make_dataset

        reset_compiler(config)
        # Triton builds its C launcher on the first kernel launch; see the note on
        # silence_python_header_warnings for why CPython's headers go first there.
        silence_python_header_warnings()
        self.dataset = make_dataset(
            self.device,
            self.dataset_config,
            feature_dim=self.model_config.feature_dim,
            pattern=self.pattern,
        )[: samples or self.dataset_config.samples]

        # Measured here, not lazily on first access.  ``reset_compiler`` throws
        # away Dynamo's compiled graphs, so a baseline that were still pending
        # when a mitigation asked for it would silently erase whatever that
        # mitigation had just warmed up -- which is exactly the mistake the
        # warm-up seeding experiment is designed to detect.
        self.eager_ms: list[float] = measure_eager(
            self.fresh_model(), self.dataset, self.device
        )

    def fresh_model(self) -> nn.Module:
        """A model with the shared, reproducible weights, in eval mode."""
        return build_model(self.model_config, seed=self._eager_seed).to(self.device).eval()

    @property
    def eager_mean_ms(self) -> float:
        return sum(self.eager_ms) / len(self.eager_ms)

    def describe(self) -> list[str]:
        return [
            f"device={self.device.type} backend={self.config.backend}",
            f"model: feature_dim={self.model_config.feature_dim}"
            f" hidden_dim={self.model_config.hidden_dim}"
            f" layers={self.model_config.message_passing_layers}",
            f"dataset: {len(self.dataset)} samples,"
            f" rows in [{self.dataset_config.min_size}, {self.dataset_config.max_size})",
            f"eager baseline mean: {self.eager_mean_ms:.2f}ms",
            f"budget: max step {self.config.budget.max_step_ms:.0f}ms,"
            f" compiled coverage >= {self.config.budget.min_compiled_ratio:.0%},"
            + (
                " no speedup required"
                if self.config.budget.min_speedup is None
                else f" min speedup {self.config.budget.min_speedup:.2f}x"
            ),
            f"cache: {cache_root(self.config)}",
        ]


def default_profile(device: torch.device, requested: Profile | None = None) -> Profile:
    """Article sizes on a GPU, shrunken sizes on a CPU."""
    if requested is not None:
        return requested
    return Profile.ARTICLE if device.type == "cuda" else Profile.SMOKE
