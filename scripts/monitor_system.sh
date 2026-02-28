#!/bin/bash
# Monitor the trading system

echo "=== Trading System Monitor ==="
echo ""

# Check agent status
echo "Agent Status:"
for agent in market_data monitoring strategy risk execution; do
  if ps -p $(cat /tmp/${agent}.pid 2>/dev/null) > /dev/null 2>&1; then
    echo "  ✓ ${agent}"
  else
    echo "  ✗ ${agent} (stopped)"
  fi
done

echo ""
echo "Recent Activity:"
echo "  Market Data: $(redis-cli XLEN stream:market_data) ticks"
echo "  Signals: $(redis-cli XLEN stream:signals) signals"
echo "  Risk Checks: $(redis-cli XLEN stream:risk_checks) checks"
echo "  Executions: $(redis-cli XLEN stream:executions) executions"

echo ""
echo "Prometheus Metrics: http://localhost:8000/metrics"
