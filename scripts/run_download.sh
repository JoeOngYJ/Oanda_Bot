#!/usr/bin/env bash
# Convenience wrapper to run the download CLI using the venv's python -m invocation.
# Usage:
#   ./scripts/run_download.sh --instrument EUR_USD --tf H1 --start 2026-01-01T00:00:00 --end 2026-01-10T00:00:00

set -euo pipefail

# If a venv is present in the project root, prefer it
VENV_DIR="$(dirname "$(dirname "$0")")/.venv"
ENV_FILE="$(dirname "$(dirname "$0")")/.env"

# If a .env file exists in the project root, source it so credentials and
# environment variables are available to the Python process. We use `set -a`
# so variables defined in the file are exported into the environment.
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
fi
if [ -x "$VENV_DIR/bin/python" ]; then
  PY_BIN="$VENV_DIR/bin/python"
else
  PY_BIN="python3"
fi

"$PY_BIN" -m oanda_trading_system.cli "$@"
