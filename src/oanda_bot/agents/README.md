# Agents Module

Runtime agent implementations:
1. `market_data/`: OANDA stream ingestion, normalization, validation, publish.
2. `strategy/`: strategy state and signal generation.
3. `risk/`: pre-trade checks, position monitoring, circuit breaker.
4. `execution/`: order creation, broker execution, fill handling.
5. `monitoring/`: health checks, metrics, alerting.

Use `python -m agents.<agent>.agent` to run each agent.

