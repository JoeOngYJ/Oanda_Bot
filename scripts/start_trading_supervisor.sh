#!/bin/bash
# Quick start script for trading process supervisor

cd /home/joe/Desktop/Algo_trading/oanda-trading-system
echo "Starting Trading Supervisor..."
echo "This service will:"
echo "  - Listen for ops commands from Discord bot (ops_control stream)"
echo "  - Start/stop trading agent processes"
echo "  - Handle sleep/wake via execution kill switch"
echo ""
echo "Required env vars:"
echo "  - REGIME_MODEL_JSON"
echo ""
echo "Press Ctrl+C to stop"
echo ""

if [ -z "$REGIME_MODEL_JSON" ]; then
  echo "Error: set REGIME_MODEL_JSON to a runtime model JSON path"
  exit 1
fi

./.venv/bin/python scripts/trading_supervisor.py \
  --project-root "${PROJECT_ROOT:-/home/joe/Desktop/Algo_trading/oanda-trading-system}" \
  --poll-seconds "${POLL_SECONDS:-2}"
