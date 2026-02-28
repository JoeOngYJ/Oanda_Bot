#!/bin/bash
# Monitor Redis streams

echo "=== Monitoring Redis Streams ==="
echo ""
echo "Market Data Stream:"
redis-cli XREAD COUNT 5 STREAMS stream:market_data 0
echo ""
echo "Signals Stream:"
redis-cli XREAD COUNT 5 STREAMS stream:signals 0
echo ""
echo "Risk Checks Stream:"
redis-cli XREAD COUNT 5 STREAMS stream:risk_checks 0
echo ""
echo "Executions Stream:"
redis-cli XREAD COUNT 5 STREAMS stream:executions 0
