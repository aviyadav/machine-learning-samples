#!/bin/sh
# Convenience launcher for the experiment suite.
#
#   ./run-cmd.sh                                  # problem + all six mitigations
#   ./run-cmd.sh problem                          # just the dilemma
#   ./run-cmd.sh solution dynamo-config           # one mitigation
#   ./run-cmd.sh all --samples 40 --backend eager
#
# Uses the project virtual environment when one is present, otherwise whatever
# Python is on PATH.  Works on Linux, macOS, WSL, and Windows via Git Bash.

set -eu

cd "$(dirname "$0")"

PY=""
for candidate in .venv/bin/python .venv/Scripts/python.exe .venv/Scripts/python; do
    if [ -x "$candidate" ]; then
        PY="$candidate"
        break
    fi
done
if [ -z "$PY" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PY=python3
    else
        PY=python
    fi
fi

# Default to the full run when no arguments were given.
if [ "$#" -eq 0 ]; then
    set -- all --profile smoke
fi

# Inductor shells out to cl.exe on Windows and needs the MSVC environment loaded
# first.  scripts/with_msvc_env.py is a transparent passthrough everywhere else,
# but invoking it only where it is needed keeps the Linux path simple.
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        exec "$PY" scripts/with_msvc_env.py -m recompilation_dilemma "$@"
        ;;
    *)
        exec "$PY" -m recompilation_dilemma "$@"
        ;;
esac
