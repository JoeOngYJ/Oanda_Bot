#!/bin/bash
# Quick start script for Market Data Agent

cd /home/joe/Desktop/Algo_trading/oanda-trading-system
echo "Starting Market Data Agent..."
echo "This agent will:"
echo "  - Connect to Oanda API"
echo "  - Stream live market data"
echo "  - Publish ticks to Redis stream"
echo ""
echo "Press Ctrl+C to stop"
echo ""
PYTHONPATH=src python -m oanda_bot.agents.market_data.agent
