#!/bin/bash
# Quick start script for Monitoring Agent

cd /home/joe/Desktop/Algo_trading/oanda-trading-system
echo "Starting Monitoring Agent..."
echo "This agent will:"
echo "  - Monitor system health"
echo "  - Collect metrics"
echo "  - Send alerts"
echo "  - Expose Prometheus metrics on :8000"
echo ""
echo "Press Ctrl+C to stop"
echo ""
python -m agents.monitoring.agent
