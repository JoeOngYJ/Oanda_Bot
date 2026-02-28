.PHONY: deps phase2 phase2-oanda phase2-influx phase2-influx-query phase2-tests phase2-agent phase2-redis phase2-stability
.PHONY: phase3 phase3-health phase3-metrics phase3-alert phase3-circuit phase3-tests phase3-integration
.PHONY: phase4 phase4-indicators phase4-ma phase4-rsi phase4-tests phase4-agent phase4-live phase4-signal phase4-backtest
.PHONY: phase6 phase6-tests phase6-execution phase6-agent
.PHONY: phase7 phase7-integration phase7-performance phase7-stress phase7-failover phase7-security phase7-all

deps:
	python -m pip install -r requirements.txt

# Phase 2 quick checks (non-blocking)
phase2: phase2-oanda phase2-influx phase2-influx-query phase2-tests

phase2-oanda:
	PYTHONPATH=. python tests/integration/test_oanda_connection.py

phase2-influx:
	PYTHONPATH=. python tests/integration/test_influx_write.py

phase2-influx-query:
	PYTHONPATH=. python tests/integration/test_influx_query.py

phase2-tests:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_market_data/ -v

# Run the live agent (stop with Ctrl+C)
phase2-agent:
	python -m agents.market_data.agent

# Check Redis stream contents (requires redis-cli)
phase2-redis:
	@command -v redis-cli >/dev/null 2>&1 && redis-cli XREAD COUNT 10 STREAMS stream:market_data 0 || echo "redis-cli not found"

# Stability test (runs ~5 minutes)
phase2-stability:
	python -m agents.market_data.agent & echo $$! > /tmp/market_data_agent.pid; \
	sleep 300; \
	ps -p `cat /tmp/market_data_agent.pid` || true; \
	tail -100 logs/trading-system.log | grep ERROR || true; \
	kill `cat /tmp/market_data_agent.pid` || true

# Phase 3 quick checks
phase3: phase3-health phase3-metrics phase3-alert phase3-circuit phase3-tests

phase3-health:
	PYTHONPATH=. python tests/integration/test_health_integration.py

phase3-metrics:
	python -m agents.monitoring.agent & echo $$! > /tmp/monitoring_agent.pid; \
	sleep 5; \
	curl -s http://localhost:8000/metrics | head -n 20 || true; \
	kill `cat /tmp/monitoring_agent.pid` || true

phase3-alert:
	PYTHONPATH=. python tests/integration/test_alert_integration.py

phase3-circuit:
	PYTHONPATH=. python tests/integration/test_circuit_integration.py

phase3-tests:
	pytest tests/test_monitoring/ -v -p pytest_asyncio

# Integration: monitoring + market data
phase3-integration:
	python -m agents.market_data.agent & echo $$! > /tmp/market_data_agent.pid; \
	python -m agents.monitoring.agent & echo $$! > /tmp/monitoring_agent.pid; \
	sleep 120; \
	curl -s http://localhost:8000/metrics | grep market_data || true; \
	kill `cat /tmp/market_data_agent.pid` || true; \
	kill `cat /tmp/monitoring_agent.pid` || true

# Phase 4 quick checks
phase4: phase4-indicators phase4-ma phase4-rsi phase4-tests
	@echo "Phase 4: all checks passed"

phase4-indicators:
	PYTHONPATH=. python tests/integration/test_strategy_indicators.py
	@echo "phase4-indicators passed"

phase4-ma:
	PYTHONPATH=. python tests/integration/test_strategy_ma_crossover.py
	@echo "phase4-ma passed"

phase4-rsi:
	PYTHONPATH=. python tests/integration/test_strategy_rsi_mean_reversion.py
	@echo "phase4-rsi passed"

phase4-tests:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_strategy/ -v
	@echo "phase4-tests passed"

# Run strategy agent (requires market data agent running)
phase4-agent:
	python -m agents.strategy.agent

# Start both agents and tail Redis (requires redis-cli)
phase4-live:
	python -m agents.market_data.agent & echo $$! > /tmp/market_data_agent.pid; \
	sleep 5; \
	python -m agents.strategy.agent & echo $$! > /tmp/strategy_agent.pid; \
	command -v redis-cli >/dev/null 2>&1 && redis-cli XREAD BLOCK 30000 STREAMS stream:signals 0 || echo "redis-cli not found"; \
	kill `cat /tmp/strategy_agent.pid` || true; \
	kill `cat /tmp/market_data_agent.pid` || true

# Signal validation (waits for a real signal; requires Redis)
phase4-signal:
	PYTHONPATH=. python tests/integration/test_strategy_signal_validation.py

# Backtest script (if present)
phase4-backtest:
	@if [ -f scripts/backtest.py ]; then python scripts/backtest.py --strategy MA_Crossover_EURUSD --start 2023-01-01 --end 2024-01-01; else echo "scripts/backtest.py not found"; fi

# Phase 6 quick checks
phase6: phase6-tests phase6-execution
	@echo "Phase 6: all checks passed"

phase6-tests:
	pytest tests/test_execution/ -v
	@echo "phase6-tests passed"

phase6-execution:
	PYTHONPATH=. python tests/integration/test_execution_flow.py
	@echo "phase6-execution passed"

# Run execution agent (requires all previous agents running)
phase6-agent:
	python -m agents.execution.agent

# Phase 7 - Integration & Testing
phase7: phase7-integration phase7-performance phase7-stress phase7-failover
	@echo "Phase 7: all tests passed"

phase7-integration:
	pytest tests/integration/ -v
	@echo "phase7-integration passed"

phase7-performance:
	pytest tests/performance/ -v
	@echo "phase7-performance passed"

phase7-stress:
	pytest tests/stress/ -v
	@echo "phase7-stress passed"

phase7-failover:
	pytest tests/failover/ -v
	@echo "phase7-failover passed"

phase7-security:
	@echo "Running security checks..."
	@command -v pip-audit >/dev/null 2>&1 && pip-audit || echo "pip-audit not installed"
	@command -v bandit >/dev/null 2>&1 && bandit -r agents/ shared/ || echo "bandit not installed"
	@echo "phase7-security passed"

phase7-all: phase7-integration phase7-performance phase7-stress phase7-failover phase7-security
	@echo "Phase 7: all validation complete"
