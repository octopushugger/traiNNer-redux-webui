#!/usr/bin/env bash
# =============================================================================
#  traiNNer-redux-webui launcher (Linux / macOS)
#
#  Edit the two variables below if you already have your own Python install
#  and/or a traiNNer-redux checkout somewhere else on your system.
#
#    PYTHON_BIN    Path to the python executable to run the web UI with.
#                  Must have the packages in webui/requirements.txt installed
#                  (pip install -r webui/requirements.txt).
#                  Defaults to the bundled python/bin/python3 next to this
#                  script.  Point this at a system/venv python if needed, e.g.:
#                      PYTHON_BIN="$HOME/venvs/trainner/bin/python"
#
#    TRAINNER_DIR  Path to your traiNNer-redux checkout
#                  (the folder that contains train.py, options/, etc.).
#                  Defaults to the traiNNer-redux/ folder next to this script.
# =============================================================================

set -euo pipefail

# Directory this script lives in (resolves symlinks).
SCRIPT_DIR="$( cd "$( dirname "$( readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}" )" )" && pwd )"

PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/venv/bin/python3}"
TRAINNER_DIR="${TRAINNER_DIR:-$SCRIPT_DIR/traiNNer-redux}"

# ── Nothing below this line normally needs editing ─────────────────────────────

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [ ! -x "$PYTHON_BIN" ]; then
    echo "[launch.sh] ERROR: Python binary not found or not executable:"
    echo "    $PYTHON_BIN"
    echo "Edit PYTHON_BIN at the top of this script (or export it) to point at"
    echo "your python executable."
    exit 1
fi

if [ ! -d "$TRAINNER_DIR" ]; then
    echo "[launch.sh] ERROR: traiNNer-redux directory not found at:"
    echo "    $TRAINNER_DIR"
    echo "Edit TRAINNER_DIR at the top of this script (or export it) to point"
    echo "at your checkout."
    exit 1
fi

export TRAINNER_REDUX_DIR="$TRAINNER_DIR"

cd "$SCRIPT_DIR/webui"
exec "$PYTHON_BIN" server.py "$@"
