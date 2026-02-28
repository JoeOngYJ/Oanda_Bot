#!/bin/bash
# Stop all trading system agents

echo "Stopping Oanda Trading System..."

for agent in market_data monitoring strategy risk execution; do
  if [ -f /tmp/${agent}.pid ]; then
    pid=$(cat /tmp/${agent}.pid)
    if ps -p $pid > /dev/null 2>&1; then
      kill $pid
      echo "✓ Stopped ${agent} agent (PID: $pid)"
    else
      echo "  ${agent} agent already stopped"
    fi
    rm -f /tmp/${agent}.pid
  fi
done

echo ""
echo "All agents stopped"
