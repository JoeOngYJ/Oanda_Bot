#!/bin/bash
# Quick start script for Risk Agent

cd /home/joe/Desktop/Algo_trading/oanda-trading-system
echo "Starting Risk Agent..."
echo "This agent will:"
echo "  - Subscribe to trade signals"
echo "  - Perform pre-trade risk checks"
echo "  - Monitor positions"
echo "  - Enforce risk limits"
echo ""
echo "Press Ctrl+C to stop"
echo ""
python -m agents.risk.agent
