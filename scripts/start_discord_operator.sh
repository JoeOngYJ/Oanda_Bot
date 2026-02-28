#!/bin/bash
# Quick start script for Discord operator bot

cd /home/joe/Desktop/Algo_trading/oanda-trading-system
echo "Starting Discord Operator Bot..."
echo "This service will:"
echo "  - Forward alerts from stream:alerts to Discord"
echo "  - Accept Discord commands for execution controls/status/profit"
echo ""
echo "Required env vars:"
echo "  - DISCORD_BOT_TOKEN"
echo "  - DISCORD_CHANNEL_ID"
echo ""
echo "Press Ctrl+C to stop"
echo ""

if [ -z "$DISCORD_BOT_TOKEN" ] || [ -z "$DISCORD_CHANNEL_ID" ]; then
  echo "Error: set DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID"
  exit 1
fi

./.venv/bin/python scripts/discord_operator_bot.py \
  --poll-seconds "${POLL_SECONDS:-3}" \
  --alert-min-severity "${ALERT_MIN_SEVERITY:-warning}" \
  --commands-prefix "${COMMANDS_PREFIX:-!}"
