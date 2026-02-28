#!/bin/bash
# Quick start script for trading journal agent

cd /home/joe/Desktop/Algo_trading/oanda-trading-system
echo "Starting Trading Journal Agent..."
echo "This service will:"
echo "  - Subscribe to execution events"
echo "  - Write daily execution logs (CSV + summary JSON)"
echo "  - Write monthly execution logs (CSV + summary JSON)"
echo ""
echo "Press Ctrl+C to stop"
echo ""

./.venv/bin/python scripts/trading_journal_agent.py \
  --output-dir "${OUTPUT_DIR:-data/reports/trading_journal}"
