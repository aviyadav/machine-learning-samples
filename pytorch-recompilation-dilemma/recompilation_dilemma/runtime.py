"""Portable runtime helpers.

The article's listing is Linux + CUDA specific in three places: ``os.sched_*``
for CPU affinity, ``os.setpriority`` for thread priority, and
``torch.cuda.synchronize()`` for timing.  Each of those is unavailable on
Windows or without a GPU, so they are wrapped here and degrade to a no-op with a
recorded reason rather than crashing the experiment.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Pick a device, honouring an explicit request when it is available."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")
    return device


def synchronize(device: torch.device) -> None:
    """Block until queued work on ``device`` has finished.

    Without this the measured interval covers kernel *launch*, not execution,
    which is why every timing helper here brackets its work with it.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class StepTimer:
    """Context manager measuring wall-clock milliseconds for one inference step."""

    __slots__ = ("_device", "_start", "elapsed_ms")

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> StepTimer:
        synchronize(self._device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        synchronize(self._device)
        self.elapsed_ms = (time.perf_counter() - self._start) * 1_000


def has_cpu_affinity() -> bool:
    """Whether this platform can pin the current thread to a CPU."""
    return hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity")


def pin_current_thread_to_one_cpu() -> tuple[int, ...]:
    """Pin the calling thread to a single CPU.  Returns the previous affinity.

    Background compilation competes with foreground inference for CPU time;
    confining it to one core is the article's first lever for reducing that
    interference.

    Affinity is per-thread on Linux, so this affects only the caller.  The call
    is still allowed to fail -- a container may expose the ``sched_*`` API and
    then refuse the request -- in which case the experiment simply records that
    the lever was unavailable (``()`` is a valid "nothing pinned" answer).
    """
    if not has_cpu_affinity():
        return ()
    try:
        allowed = os.sched_getaffinity(0)
        if len(allowed) < 2:
            # Already confined; nothing to gain and nothing to restore.
            return ()
        os.sched_setaffinity(0, {min(allowed)})
    except OSError:
        return ()
    return tuple(sorted(allowed))


def restore_cpu_affinity(previous: tuple[int, ...]) -> None:
    if previous and has_cpu_affinity():
        try:
            os.sched_setaffinity(0, set(previous))
        except OSError:
            return


def has_thread_priority() -> bool:
    return hasattr(os, "setpriority") and hasattr(os, "PRIO_PROCESS")


def lower_current_thread_priority(level: int = 10) -> bool:
    """Deprioritise the calling thread.  Returns whether it took effect.

    Lowering priority only ever needs an increased nice value, which any user may
    do; raising it is what requires privilege.  ``PRIO_PROCESS`` with a native
    thread id targets this thread alone on Linux.
    """
    if not has_thread_priority():
        return False
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), level)
    except OSError:  # includes PermissionError
        return False
    return True


def native_thread_id() -> int:
    return threading.get_native_id()


@dataclass
class InterferenceControls:
    """Which interference-reduction levers actually applied on this platform.

    The async-compilation mitigation is only as good as these levers, so the
    report records them instead of silently succeeding on a platform that
    ignores them.
    """

    pinned_cpu: bool = False
    lowered_priority: bool = False
    single_compile_thread: bool = False
    gil_interval_ms: float | None = None
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        applied = [
            name
            for name, ok in (
                ("cpu affinity", self.pinned_cpu),
                ("thread priority", self.lowered_priority),
                ("single compile thread", self.single_compile_thread),
            )
            if ok
        ]
        if self.gil_interval_ms is not None:
            applied.append("gil interval")
        if not applied:
            return "none applied on this platform"
        return ", ".join(applied)
