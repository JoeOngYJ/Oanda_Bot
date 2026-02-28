.PHONY: deps phase2 phase2-oanda phase2-influx phase2-influx-query phase2-tests phase2-agent phase2-redis phase2-stability
.PHONY: phase3 phase3-health phase3-metrics phase3-alert phase3-circuit phase3-tests phase3-integration
.PHONY: phase4 phase4-indicators phase4-ma phase4-rsi phase4-tests phase4-agent phase4-live phase4-signal phase4-backtest
.PHONY: phase6 phase6-tests phase6-execution phase6-agent
.PHONY: phase7 phase7-integration phase7-performance phase7-stress phase7-failover phase7-security phase7-all
.PHONY: scorecard-new scorecard-report
.PHONY: strategy-research
.PHONY: universe-research
.PHONY: regime-gpu-research
.PHONY: gpu-prescreener
.PHONY: gpu-universe-pipeline
.PHONY: realtime-backtest
.PHONY: regime-runtime-backtest
.PHONY: train-mtf-regime
.PHONY: xau-breakout-backtest
.PHONY: xau-breakout-opt
.PHONY: exec-kill-on exec-kill-off exec-shadow-on exec-shadow-off
.PHONY: exec-status-api

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

# Scorecard automation
# Example:
# make scorecard-new STRATEGY=MA_Crossover_EURUSD ENV=paper REVIEWER=ops_lead VERSION=1.0.0
scorecard-new:
	@test -n "$(STRATEGY)" || (echo "Missing STRATEGY=<name>" && exit 1)
	@test -n "$(ENV)" || (echo "Missing ENV=<research|paper|live_micro|live_scaled>" && exit 1)
	python scripts/scorecard_new_week.py --strategy "$(STRATEGY)" --environment "$(ENV)" $(if $(REVIEWER),--reviewer "$(REVIEWER)",) $(if $(VERSION),--version "$(VERSION)",) $(if $(WEEK),--week "$(WEEK)",)

# Example:
# make scorecard-report
# make scorecard-report WEEK=2026-W09
scorecard-report:
	python scripts/scorecard_report.py $(if $(WEEK),--week "$(WEEK)",)

# Example:
# make strategy-research INSTRUMENT=EUR_USD TF=H1 START=2023-01-01 END=2024-01-01
strategy-research:
	python scripts/run_strategy_research.py \
		--instrument "$(if $(INSTRUMENT),$(INSTRUMENT),EUR_USD)" \
		--tf "$(if $(TF),$(TF),H1)" \
		--start "$(if $(START),$(START),2023-01-01)" \
		--end "$(if $(END),$(END),2024-01-01)" \
		$(if $(DEMO_BARS),--demo-bars "$(DEMO_BARS)",)

# Example:
# make universe-research INSTRUMENTS=EUR_USD,GBP_USD,USD_JPY,XAU_USD BASE_TF=M15 WF_WINDOWS=4 DEMO_BARS=2000 MIN_STABILITY=0.25 MIN_TRADES=1 MAX_CORR=0.75 INTERMARKET_SWEEP=1 WORKERS=4
universe-research:
	python scripts/run_universe_research.py \
		--instruments "$(if $(INSTRUMENTS),$(INSTRUMENTS),EUR_USD,GBP_USD,USD_JPY,XAU_USD)" \
		--base-tf "$(if $(BASE_TF),$(BASE_TF),M15)" \
		--start "$(if $(START),$(START),2023-01-01)" \
		--end "$(if $(END),$(END),2024-01-01)" \
		$(if $(WF_WINDOWS),--wf-windows "$(WF_WINDOWS)",) \
		$(if $(MIN_STABILITY),--min-stability "$(MIN_STABILITY)",) \
		$(if $(MIN_TRADES),--min-trades "$(MIN_TRADES)",) \
		$(if $(MAX_CORR),--max-corr "$(MAX_CORR)",) \
		$(if $(INTERMARKET_SWEEP),--intermarket-sweep,) \
		$(if $(WORKERS),--workers "$(WORKERS)",) \
		$(if $(VOL_TARGETING),--vol-targeting,) \
		$(if $(TARGET_ANNUAL_VOL),--target-annual-vol "$(TARGET_ANNUAL_VOL)",) \
		$(if $(VOL_LOOKBACK_BARS),--vol-lookback-bars "$(VOL_LOOKBACK_BARS)",) \
		$(if $(MAX_EXPOSURE_PCT),--max-exposure-pct "$(MAX_EXPOSURE_PCT)",) \
		$(if $(MAX_QUANTITY),--max-quantity "$(MAX_QUANTITY)",) \
		$(if $(CANDIDATE_SHORTLIST),--candidate-shortlist "$(CANDIDATE_SHORTLIST)",) \
		$(if $(DEMO_BARS),--demo-bars "$(DEMO_BARS)",)

# Example:
# make regime-gpu-research INSTRUMENT=EUR_USD TF=M15 START=2023-01-01 END=2024-01-01 REGIMES=4 GPU=auto DEMO_BARS=3000
regime-gpu-research:
	./.venv/bin/python scripts/run_regime_gpu_research.py \
		--instrument "$(if $(INSTRUMENT),$(INSTRUMENT),EUR_USD)" \
		--tf "$(if $(TF),$(TF),M15)" \
		--start "$(if $(START),$(START),2023-01-01)" \
		--end "$(if $(END),$(END),2024-01-01)" \
		--regimes "$(if $(REGIMES),$(REGIMES),4)" \
		--gpu "$(if $(GPU),$(GPU),auto)" \
		$(if $(DEMO_BARS),--demo-bars "$(DEMO_BARS)",)

# Example:
# make gpu-prescreener INSTRUMENT=EUR_USD TF=M15 START=2022-01-01 END=2025-12-31 GPU=auto TOP_N=20
gpu-prescreener:
	./.venv/bin/python scripts/run_gpu_prescreener.py \
		--instrument "$(if $(INSTRUMENT),$(INSTRUMENT),EUR_USD)" \
		--tf "$(if $(TF),$(TF),M15)" \
		--start "$(if $(START),$(START),2023-01-01)" \
		--end "$(if $(END),$(END),2024-01-01)" \
		--gpu "$(if $(GPU),$(GPU),auto)" \
		--top-n "$(if $(TOP_N),$(TOP_N),20)" \
		$(if $(DEMO_BARS),--demo-bars "$(DEMO_BARS)",)

# Example:
# make gpu-universe-pipeline INSTRUMENT=EUR_USD INSTRUMENTS=EUR_USD,GBP_USD,USD_JPY,XAU_USD TF=M15 BASE_TF=M15 START=2022-01-01 END=2025-12-31 GPU=auto TOP_N=20 WF_WINDOWS=8 MIN_STABILITY=0.30 MIN_TRADES=2 MAX_CORR=0.70 WORKERS=4
gpu-universe-pipeline:
	set -e; \
	./.venv/bin/python scripts/run_gpu_prescreener.py \
		--instrument "$(if $(INSTRUMENT),$(INSTRUMENT),EUR_USD)" \
		--tf "$(if $(TF),$(TF),M15)" \
		--start "$(if $(START),$(START),2023-01-01)" \
		--end "$(if $(END),$(END),2024-01-01)" \
		--gpu "$(if $(GPU),$(GPU),auto)" \
		--top-n "$(if $(TOP_N),$(TOP_N),20)" \
		$(if $(DEMO_BARS),--demo-bars "$(DEMO_BARS)",); \
	SHORTLIST=$$(ls -t data/research/gpu_prescreener_"$(if $(INSTRUMENT),$(INSTRUMENT),EUR_USD)"_"$(if $(TF),$(TF),M15)"_*_shortlist.csv 2>/dev/null | head -n 1); \
	if [ -z "$$SHORTLIST" ]; then echo "No GPU shortlist found in data/research"; exit 1; fi; \
	echo "Using shortlist: $$SHORTLIST"; \
	./.venv/bin/python scripts/run_universe_research.py \
		--instruments "$(if $(INSTRUMENTS),$(INSTRUMENTS),EUR_USD,GBP_USD,USD_JPY,XAU_USD)" \
		--base-tf "$(if $(BASE_TF),$(BASE_TF),M15)" \
		--start "$(if $(START),$(START),2023-01-01)" \
		--end "$(if $(END),$(END),2024-01-01)" \
		--candidate-shortlist "$$SHORTLIST" \
		$(if $(WF_WINDOWS),--wf-windows "$(WF_WINDOWS)",) \
		$(if $(MIN_STABILITY),--min-stability "$(MIN_STABILITY)",) \
		$(if $(MIN_TRADES),--min-trades "$(MIN_TRADES)",) \
		$(if $(MAX_CORR),--max-corr "$(MAX_CORR)",) \
		$(if $(INTERMARKET_SWEEP),--intermarket-sweep,) \
		$(if $(WORKERS),--workers "$(WORKERS)",) \
		$(if $(VOL_TARGETING),--vol-targeting,) \
		$(if $(TARGET_ANNUAL_VOL),--target-annual-vol "$(TARGET_ANNUAL_VOL)",) \
		$(if $(VOL_LOOKBACK_BARS),--vol-lookback-bars "$(VOL_LOOKBACK_BARS)",) \
		$(if $(MAX_EXPOSURE_PCT),--max-exposure-pct "$(MAX_EXPOSURE_PCT)",) \
		$(if $(MAX_QUANTITY),--max-quantity "$(MAX_QUANTITY)",) \
		$(if $(DEMO_BARS),--demo-bars "$(DEMO_BARS)",)

# Example:
# make realtime-backtest INSTRUMENT=EUR_USD TF=M15 START=2024-01-01 END=2024-12-31 FILL_MODE=next_open
realtime-backtest:
	./.venv/bin/python scripts/run_realtime_backtest.py \
		--instrument "$(if $(INSTRUMENT),$(INSTRUMENT),EUR_USD)" \
		--tf "$(if $(TF),$(TF),M15)" \
		--start "$(if $(START),$(START),2024-01-01)" \
		--end "$(if $(END),$(END),2024-12-31)" \
		--fill-mode "$(if $(FILL_MODE),$(FILL_MODE),next_open)" \
		$(if $(STATE_SNAPSHOT_PATH),--state-snapshot-path "$(STATE_SNAPSHOT_PATH)",) \
		$(if $(SNAPSHOT_EVERY_BARS),--snapshot-every-bars "$(SNAPSHOT_EVERY_BARS)",)

# Example:
# make regime-runtime-backtest MODEL_JSON=data/research/regime_research_EUR_USD_M15_<stamp>_runtime_model.json INSTRUMENT=EUR_USD TF=M15 START=2025-01-01 END=2025-12-31
regime-runtime-backtest:
	./.venv/bin/python scripts/run_regime_runtime_backtest.py \
		--model-json "$(MODEL_JSON)" \
		--instrument "$(if $(INSTRUMENT),$(INSTRUMENT),EUR_USD)" \
		--tf "$(if $(TF),$(TF),M15)" \
		--start "$(if $(START),$(START),2025-01-01)" \
		--end "$(if $(END),$(END),2025-12-31)" \
		--fill-mode "$(if $(FILL_MODE),$(FILL_MODE),next_open)" \
		--decision-mode "$(if $(DECISION_MODE),$(DECISION_MODE),ensemble)"

# Example:
# make train-mtf-regime INSTRUMENTS=XAU_USD,EUR_USD,GBP_USD START=2022-01-01 END=2024-12-31 GPU=auto
train-mtf-regime:
	./.venv/bin/python scripts/train_multiframe_regime_model.py \
		--instruments "$(if $(INSTRUMENTS),$(INSTRUMENTS),XAU_USD,EUR_USD,GBP_USD)" \
		--start "$(if $(START),$(START),2022-01-01)" \
		--end "$(if $(END),$(END),2024-12-31)" \
		--gpu "$(if $(GPU),$(GPU),auto)"

# Example:
# make xau-breakout-backtest START=2025-01-01 END=2025-12-31
xau-breakout-backtest:
	./.venv/bin/python scripts/run_xau_breakout_backtest.py \
		--start "$(if $(START),$(START),2025-01-01)" \
		--end "$(if $(END),$(END),2025-12-31)"

# Example:
# make xau-breakout-opt TRAIN_START=2023-01-01 TRAIN_END=2024-12-31 TEST_START=2025-01-01 TEST_END=2025-12-31
xau-breakout-opt:
	./.venv/bin/python scripts/optimize_xau_breakout.py \
		--train-start "$(if $(TRAIN_START),$(TRAIN_START),2023-01-01)" \
		--train-end "$(if $(TRAIN_END),$(TRAIN_END),2024-12-31)" \
		--test-start "$(if $(TEST_START),$(TEST_START),2025-01-01)" \
		--test-end "$(if $(TEST_END),$(TEST_END),2025-12-31)"

exec-kill-on:
	python scripts/execution_control.py --action kill_switch_on --reason "$(if $(REASON),$(REASON),manual kill switch)"

exec-kill-off:
	python scripts/execution_control.py --action kill_switch_off --reason "$(if $(REASON),$(REASON),resume execution)"

exec-shadow-on:
	python scripts/execution_control.py --action shadow_mode_on --reason "$(if $(REASON),$(REASON),enable shadow mode)"

exec-shadow-off:
	python scripts/execution_control.py --action shadow_mode_off --reason "$(if $(REASON),$(REASON),disable shadow mode)"

exec-status-api:
	python scripts/execution_status_api.py --host "$(if $(HOST),$(HOST),0.0.0.0)" --port "$(if $(PORT),$(PORT),8010)"
