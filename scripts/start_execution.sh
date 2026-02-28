#!/bin/bash
# Quick start script for Execution Agent

cd /home/joe/Desktop/Algo_trading/oanda-trading-system
echo "Starting Execution Agent..."
echo "This agent will:"
echo "  - Subscribe to approved signals"
echo "  - Create orders"
echo "  - Execute on Oanda"
echo "  - Track fills"
echo ""
echo "Press Ctrl+C to stop"
echo ""
python -m agents.execution.agent
