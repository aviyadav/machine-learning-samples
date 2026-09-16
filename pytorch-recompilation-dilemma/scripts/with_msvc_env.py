"""Run a command with the MSVC build environment loaded (Windows only).

``torch.compile``'s Inductor backend shells out to ``cl.exe`` and inherits
whatever environment the Python process already has -- it does *not* locate
Visual Studio on its own.  On Windows that means compiling anything only works
from a "Developer Command Prompt", or from any shell where ``vcvars64.bat`` has
been sourced first.

This helper bridges that gap: it captures the environment that ``vcvars64.bat``
produces and re-executes the requested command with it, so a plain
``python scripts/with_msvc_env.py -m recompilation_dilemma ...`` works.

On Linux and macOS it is a transparent passthrough -- the toolchain is already
on ``PATH`` there.

Usage::

    python scripts/with_msvc_env.py -m recompilation_dilemma problem --profile smoke
    python scripts/with_msvc_env.py -m recompilation_dilemma all --profile smoke
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

VSWHERE = (
    Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
    / "Microsoft Visual Studio"
    / "Installer"
    / "vswhere.exe"
)


def _vcvars_bat() -> Path | None:
    if not VSWHERE.exists():
        return None
    result = subprocess.run(
        [str(VSWHERE), "-latest", "-products", "*", "-property", "installationPath"],
        capture_output=True,
        text=True,
    )
    install = result.stdout.strip().splitlines()
    if not install:
        return None

    # Prefer the native x64 toolchain, fall back to the x86-hosted one.
    for relative in (
        r"VC\Auxiliary\Build\vcvars64.bat",
        r"VC\Auxiliary\Build\vcvarsall.bat",
    ):
        candidate = Path(install[0]) / relative
        if candidate.exists():
            return candidate
    return None


def capture_msvc_environment(bat: Path) -> dict[str, str]:
    """Return the environment variables that ``bat`` defines.

    ``set`` inside the same ``cmd`` invocation is the only reliable way to read
    back the result: ``vcvars`` mutates the environment of its own shell rather
    than returning anything.
    """
    script = Path(tempfile.gettempdir()) / "recompilation_dilemma_msvc_env.bat"
    script.write_text(f'@echo off\ncall "{bat}" >nul\nset\n', encoding="ascii")

    result = subprocess.run(["cmd.exe", "/c", str(script)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"vcvars failed ({result.returncode}): {result.stderr.strip()}")

    environment: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            environment[key.strip().upper()] = value.strip()
    return environment


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    if sys.platform != "win32":
        return subprocess.run([sys.executable, *argv]).returncode

    bat = _vcvars_bat()
    if bat is None:
        print(
            "No Visual Studio installation found (vswhere returned nothing).\n"
            "Run the command from a shell that already has cl.exe on PATH; the "
            "Inductor backend needs it.",
            file=sys.stderr,
        )
        return subprocess.run([sys.executable, *argv]).returncode

    try:
        environment = capture_msvc_environment(bat)
    except RuntimeError as error:
        print(f"Could not load the MSVC environment: {error}", file=sys.stderr)
        return 1

    merged = {**os.environ, **environment}
    print(f"[with_msvc_env] loaded {bat.name} ({len(environment)} variables)", file=sys.stderr)
    return subprocess.run([sys.executable, *argv], env=merged).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
