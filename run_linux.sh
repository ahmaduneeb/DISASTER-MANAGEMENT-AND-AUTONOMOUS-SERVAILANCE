#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# AEGIS - UAV Strategy Engine
# Linux / macOS launcher: creates a virtual environment, installs the
# requirements, and starts the browser dashboard.
#
# Usage:
#   ./run_linux.sh
#   ./run_linux.sh --port 8000 --no-browser
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"

echo "============================================================"
echo "  AEGIS - UAV Strategy Engine"
echo "  Disaster Management & Autonomous Surveillance"
echo "============================================================"
echo

# ---------------------------------------------------------------------------
# 1. Locate Python 3.10+
# ---------------------------------------------------------------------------
PYTHON=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)' >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  echo "[ERROR] Python 3.10 or newer was not found." >&2
  echo "        Install it with: sudo apt install python3 python3-venv python3-pip" >&2
  exit 1
fi

echo "[1/3] Using $($PYTHON --version)"

# ---------------------------------------------------------------------------
# 2. Create / reuse the virtual environment
# ---------------------------------------------------------------------------
if [ ! -x ".venv/bin/python" ]; then
  echo "[2/3] Creating virtual environment in .venv ..."
  "$PYTHON" -m venv .venv
else
  echo "[2/3] Reusing existing virtual environment."
fi

VENV_PY=".venv/bin/python"

# ---------------------------------------------------------------------------
# 3. Install dependencies and launch
# ---------------------------------------------------------------------------
echo "[3/3] Installing dependencies from requirements.txt ..."
"$VENV_PY" -m pip install --upgrade pip >/dev/null 2>&1 || true
"$VENV_PY" -m pip install -r requirements.txt

echo
echo "  URL:  http://127.0.0.1:5000"
echo "  Press CTRL+C to stop the server."
echo

exec "$VENV_PY" app.py "$@"
