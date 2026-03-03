#!/bin/bash
# Quick start script for Discord execution notifier bot

cd /home/joe/Desktop/Algo_trading/oanda-trading-system
echo "Starting Discord Execution Notifier Bot..."
echo "This service will:"
echo "  - Subscribe to stream:executions"
echo "  - Send order placed/closed updates to Discord"
echo "  - Include current OANDA account balance/NAV"
echo ""
echo "Required env vars:"
echo "  - DISCORD_EXEC_BOT_TOKEN (or DISCORD_BOT_TOKEN fallback)"
echo "Optional env vars:"
echo "  - DISCORD_EXEC_CHANNEL_ID (fallback to DISCORD_CHANNEL_ID, default: 1477609642258337954)"
echo "  - BALANCE_TIMEOUT_SECONDS (default: 10)"
echo ""
echo "Press Ctrl+C to stop"
echo ""

if [ -z "${DISCORD_EXEC_BOT_TOKEN:-}" ] && [ -z "${DISCORD_BOT_TOKEN:-}" ]; then
  echo "Error: set DISCORD_EXEC_BOT_TOKEN or DISCORD_BOT_TOKEN"
  exit 1
fi

./.venv/bin/python scripts/discord_execution_notifier.py \
  --balance-timeout-seconds "${BALANCE_TIMEOUT_SECONDS:-10}"
