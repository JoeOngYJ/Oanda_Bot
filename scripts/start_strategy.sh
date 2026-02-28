#!/bin/bash
# Quick start script for Strategy Agent

cd /home/joe/Desktop/Algo_trading/oanda-trading-system
echo "Starting Strategy Agent..."
echo "This agent will:"
echo "  - Subscribe to market data"
echo "  - Calculate technical indicators"
echo "  - Generate trade signals"
echo "  - Publish signals to Redis stream"
echo ""
echo "Press Ctrl+C to stop"
echo ""
python -m agents.strategy.agent
