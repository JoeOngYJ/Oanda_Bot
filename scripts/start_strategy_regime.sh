#!/bin/bash
# Quick start script for Regime Runtime Strategy Agent

cd /home/joe/Desktop/Algo_trading/oanda-trading-system
echo "Starting Regime Runtime Strategy Agent..."
echo "This agent will:"
echo "  - Subscribe to market data ticks"
echo "  - Build multi-timeframe OHLCV bars"
echo "  - Predict market regime from model JSON"
echo "  - Publish trade signals to Redis stream"
echo ""
echo "Usage:"
echo "  MODEL_JSON=data/research/<runtime_model>.json ./scripts/start_strategy_regime.sh"
echo ""
echo "Press Ctrl+C to stop"
echo ""
if [ -z "$MODEL_JSON" ]; then
  echo "Error: set MODEL_JSON to a runtime model JSON path"
  exit 1
fi

PYTHONPATH=src ./.venv/bin/python -m oanda_bot.agents.strategy.regime_runtime_agent \
  --model-json "$MODEL_JSON" \
  --instrument "${INSTRUMENT:-XAU_USD}" \
  --decision-mode "${DECISION_MODE:-ensemble}" \
  --quantity "${QUANTITY:-2}" \
  --min-confidence "${MIN_CONFIDENCE:-0.25}" \
  --warmup "${WARMUP:-on}" \
  --warmup-base-bars "${WARMUP_BASE_BARS:-1500}" \
  ${WARMUP_M15_BARS:+--warmup-m15-bars "$WARMUP_M15_BARS"} \
  ${WARMUP_H1_BARS:+--warmup-h1-bars "$WARMUP_H1_BARS"} \
  ${WARMUP_H4_BARS:+--warmup-h4-bars "$WARMUP_H4_BARS"} \
  ${WARMUP_D1_BARS:+--warmup-d1-bars "$WARMUP_D1_BARS"} \
  --gpu "${GPU:-auto}"
