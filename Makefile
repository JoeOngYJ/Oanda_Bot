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
.PHONY: mtf-prepare-data mtf-train mtf-eval mtf-pipeline mtf-clean
.PHONY: xau-mtf-dev
.PHONY: xau-multi-session-train xau-multi-session-smoke
.PHONY: xau-breakout-backtest
.PHONY: xau-breakout-opt
.PHONY: exec-kill-on exec-kill-off exec-shadow-on exec-shadow-off
.PHONY: exec-status-api
.PHONY: strategy-regime-agent regime-shadow-smoke strategy-regime-agent-xau-prod
.PHONY: discord-operator-bot
.PHONY: trading-supervisor
.PHONY: trading-journal-agent
.PHONY: systemd-user-install systemd-user-enable systemd-user-status
.PHONY: systemd-user-enable-infra

deps:
	python -m pip install -r requirements.txt

# Phase 2 quick checks (non-blocking)
phase2: phase2-oanda phase2-influx phase2-influx-query phase2-tests

phase2-oanda:
	PYTHONPATH=src python tests/integration/test_oanda_connection.py

phase2-influx:
	PYTHONPATH=src python tests/integration/test_influx_write.py

phase2-influx-query:
	PYTHONPATH=src python tests/integration/test_influx_query.py

phase2-tests:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_market_data/ -v

# Run the live agent (stop with Ctrl+C)
phase2-agent:
	python -m oanda_bot.agents.market_data.agent

# Check Redis stream contents (requires redis-cli)
phase2-redis:
	@command -v redis-cli >/dev/null 2>&1 && redis-cli XREAD COUNT 10 STREAMS stream:market_data 0 || echo "redis-cli not found"

# Stability test (runs ~5 minutes)
phase2-stability:
	python -m oanda_bot.agents.market_data.agent & echo $$! > /tmp/market_data_agent.pid; \
	sleep 300; \
	ps -p `cat /tmp/market_data_agent.pid` || true; \
	tail -100 logs/trading-system.log | grep ERROR || true; \
	kill `cat /tmp/market_data_agent.pid` || true

# Phase 3 quick checks
phase3: phase3-health phase3-metrics phase3-alert phase3-circuit phase3-tests

phase3-health:
	PYTHONPATH=src python tests/integration/test_health_integration.py

phase3-metrics:
	python -m oanda_bot.agents.monitoring.agent & echo $$! > /tmp/monitoring_agent.pid; \
	sleep 5; \
	curl -s http://localhost:8000/metrics | head -n 20 || true; \
	kill `cat /tmp/monitoring_agent.pid` || true

phase3-alert:
	PYTHONPATH=src python tests/integration/test_alert_integration.py

phase3-circuit:
	PYTHONPATH=src python tests/integration/test_circuit_integration.py

phase3-tests:
	pytest tests/test_monitoring/ -v -p pytest_asyncio

# Integration: monitoring + market data
phase3-integration:
	python -m oanda_bot.agents.market_data.agent & echo $$! > /tmp/market_data_agent.pid; \
	python -m oanda_bot.agents.monitoring.agent & echo $$! > /tmp/monitoring_agent.pid; \
	sleep 120; \
	curl -s http://localhost:8000/metrics | grep market_data || true; \
	kill `cat /tmp/market_data_agent.pid` || true; \
	kill `cat /tmp/monitoring_agent.pid` || true

# Phase 4 quick checks
phase4: phase4-indicators phase4-ma phase4-rsi phase4-tests
	@echo "Phase 4: all checks passed"

phase4-indicators:
	PYTHONPATH=src python tests/integration/test_strategy_indicators.py
	@echo "phase4-indicators passed"

phase4-ma:
	PYTHONPATH=src python tests/integration/test_strategy_ma_crossover.py
	@echo "phase4-ma passed"

phase4-rsi:
	PYTHONPATH=src python tests/integration/test_strategy_rsi_mean_reversion.py
	@echo "phase4-rsi passed"

phase4-tests:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_strategy/ -v
	@echo "phase4-tests passed"

# Run strategy agent (requires market data agent running)
phase4-agent:
	python -m oanda_bot.agents.strategy.agent

# Start both agents and tail Redis (requires redis-cli)
phase4-live:
	python -m oanda_bot.agents.market_data.agent & echo $$! > /tmp/market_data_agent.pid; \
	sleep 5; \
	python -m oanda_bot.agents.strategy.agent & echo $$! > /tmp/strategy_agent.pid; \
	command -v redis-cli >/dev/null 2>&1 && redis-cli XREAD BLOCK 30000 STREAMS stream:signals 0 || echo "redis-cli not found"; \
	kill `cat /tmp/strategy_agent.pid` || true; \
	kill `cat /tmp/market_data_agent.pid` || true

# Signal validation (waits for a real signal; requires Redis)
phase4-signal:
	PYTHONPATH=src python tests/integration/test_strategy_signal_validation.py

# Example:
# make strategy-regime-agent MODEL_JSON=data/research/multiframe_regime_model_latest.json INSTRUMENT=XAU_USD DECISION_MODE=ensemble GPU=auto
strategy-regime-agent:
	./.venv/bin/python -m oanda_bot.agents.strategy.regime_runtime_agent \
		--model-json "$(MODEL_JSON)" \
		--instrument "$(if $(INSTRUMENT),$(INSTRUMENT),XAU_USD)" \
		--decision-mode "$(if $(DECISION_MODE),$(DECISION_MODE),ensemble)" \
		--quantity "$(if $(QUANTITY),$(QUANTITY),2)" \
		--min-confidence "$(if $(MIN_CONFIDENCE),$(MIN_CONFIDENCE),0.25)" \
		--warmup "$(if $(WARMUP),$(WARMUP),on)" \
		--warmup-base-bars "$(if $(WARMUP_BASE_BARS),$(WARMUP_BASE_BARS),1500)" \
		$(if $(WARMUP_M15_BARS),--warmup-m15-bars "$(WARMUP_M15_BARS)",) \
		$(if $(WARMUP_H1_BARS),--warmup-h1-bars "$(WARMUP_H1_BARS)",) \
		$(if $(WARMUP_H4_BARS),--warmup-h4-bars "$(WARMUP_H4_BARS)",) \
		$(if $(WARMUP_D1_BARS),--warmup-d1-bars "$(WARMUP_D1_BARS)",) \
		--gpu "$(if $(GPU),$(GPU),auto)"

# XAU production profile with deeper multi-timeframe warmup context.
# Example:
# make strategy-regime-agent-xau-prod MODEL_JSON=models/active/multiframe_regime_model_20260228_194306.json
strategy-regime-agent-xau-prod:
	@$(MAKE) strategy-regime-agent \
		MODEL_JSON="$(MODEL_JSON)" \
		INSTRUMENT="$(if $(INSTRUMENT),$(INSTRUMENT),XAU_USD)" \
		DECISION_MODE="$(if $(DECISION_MODE),$(DECISION_MODE),ensemble)" \
		QUANTITY="$(if $(QUANTITY),$(QUANTITY),3)" \
		MIN_CONFIDENCE="$(if $(MIN_CONFIDENCE),$(MIN_CONFIDENCE),0.25)" \
		GPU="$(if $(GPU),$(GPU),auto)" \
		WARMUP=on \
		WARMUP_M15_BARS="$(if $(WARMUP_M15_BARS),$(WARMUP_M15_BARS),5000)" \
		WARMUP_H1_BARS="$(if $(WARMUP_H1_BARS),$(WARMUP_H1_BARS),3000)" \
		WARMUP_H4_BARS="$(if $(WARMUP_H4_BARS),$(WARMUP_H4_BARS),1500)" \
		WARMUP_D1_BARS="$(if $(WARMUP_D1_BARS),$(WARMUP_D1_BARS),750)"

# Smoke test: run execution in shadow mode and keep kill-switch off to prevent live sends.
# Requires local Redis and valid OANDA env vars for market data streaming.
regime-shadow-smoke:
	@test -n "$(MODEL_JSON)" || (echo "Missing MODEL_JSON=<runtime_model.json>" && exit 1)
	cd /home/joe/Desktop/Algo_trading/oanda-trading-system && \
		EXECUTION_SHADOW_MODE=true EXECUTION_LIVE_ENABLED=false ./.venv/bin/python -m oanda_bot.agents.market_data.agent & echo $$! > /tmp/market_data_agent.pid; \
		sleep 3; \
		EXECUTION_SHADOW_MODE=true EXECUTION_LIVE_ENABLED=false ./.venv/bin/python -m oanda_bot.agents.risk.agent & echo $$! > /tmp/risk_agent.pid; \
		sleep 2; \
		EXECUTION_SHADOW_MODE=true EXECUTION_LIVE_ENABLED=false ./.venv/bin/python -m oanda_bot.agents.execution.agent & echo $$! > /tmp/execution_agent.pid; \
		sleep 2; \
		EXECUTION_SHADOW_MODE=true EXECUTION_LIVE_ENABLED=false ./.venv/bin/python -m oanda_bot.agents.strategy.regime_runtime_agent --model-json "$(MODEL_JSON)" --instrument "$(if $(INSTRUMENT),$(INSTRUMENT),XAU_USD)" --decision-mode "$(if $(DECISION_MODE),$(DECISION_MODE),ensemble)" --warmup "$(if $(WARMUP),$(WARMUP),on)" --warmup-base-bars "$(if $(WARMUP_BASE_BARS),$(WARMUP_BASE_BARS),1500)" $(if $(WARMUP_M15_BARS),--warmup-m15-bars "$(WARMUP_M15_BARS)",) $(if $(WARMUP_H1_BARS),--warmup-h1-bars "$(WARMUP_H1_BARS)",) $(if $(WARMUP_H4_BARS),--warmup-h4-bars "$(WARMUP_H4_BARS)",) $(if $(WARMUP_D1_BARS),--warmup-d1-bars "$(WARMUP_D1_BARS)",) --gpu "$(if $(GPU),$(GPU),auto)" & echo $$! > /tmp/strategy_regime_agent.pid; \
		sleep "$(if $(DURATION_SEC),$(DURATION_SEC),60)"; \
		kill `cat /tmp/strategy_regime_agent.pid` || true; \
		kill `cat /tmp/execution_agent.pid` || true; \
		kill `cat /tmp/risk_agent.pid` || true; \
		kill `cat /tmp/market_data_agent.pid` || true

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
	PYTHONPATH=src python tests/integration/test_execution_flow.py
	@echo "phase6-execution passed"

# Run execution agent (requires all previous agents running)
phase6-agent:
	python -m oanda_bot.agents.execution.agent

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
# make regime-runtime-backtest MODEL_JSON=data/research/regime_research_EUR_USD_M15_<stamp>_runtime_model.json INSTRUMENT=EUR_USD TF=M15 START=2025-01-01 END=2025-12-31 GPU=auto
regime-runtime-backtest:
	./.venv/bin/python scripts/run_regime_runtime_backtest.py \
		--model-json "$(MODEL_JSON)" \
		--instrument "$(if $(INSTRUMENT),$(INSTRUMENT),EUR_USD)" \
		--tf "$(if $(TF),$(TF),M15)" \
		--start "$(if $(START),$(START),2025-01-01)" \
		--end "$(if $(END),$(END),2025-12-31)" \
		--fill-mode "$(if $(FILL_MODE),$(FILL_MODE),next_open)" \
		--decision-mode "$(if $(DECISION_MODE),$(DECISION_MODE),ensemble)" \
		--gpu "$(if $(GPU),$(GPU),auto)" \
		--risk-per-trade-pct "$(if $(RISK_PER_TRADE_PCT),$(RISK_PER_TRADE_PCT),0.01)" \
		--max-notional-exposure-pct "$(if $(MAX_NOTIONAL_EXPOSURE_PCT),$(MAX_NOTIONAL_EXPOSURE_PCT),1.0)" \
		--min-quantity "$(if $(MIN_QUANTITY),$(MIN_QUANTITY),1)" \
		--max-quantity "$(if $(MAX_QUANTITY),$(MAX_QUANTITY),100000)" \
		--max-drawdown-stop-pct "$(if $(MAX_DRAWDOWN_STOP_PCT),$(MAX_DRAWDOWN_STOP_PCT),0.20)" \
		--daily-loss-limit-pct "$(if $(DAILY_LOSS_LIMIT_PCT),$(DAILY_LOSS_LIMIT_PCT),0.05)" \
		--financing "$(if $(FINANCING),$(FINANCING),on)" \
		--default-financing-long-rate "$(if $(DEFAULT_FINANCING_LONG_RATE),$(DEFAULT_FINANCING_LONG_RATE),0.03)" \
		--default-financing-short-rate "$(if $(DEFAULT_FINANCING_SHORT_RATE),$(DEFAULT_FINANCING_SHORT_RATE),0.03)" \
		--rollover-hour-utc "$(if $(ROLLOVER_HOUR_UTC),$(ROLLOVER_HOUR_UTC),22)" \
		$(if $(STRATEGY_PARAMS_CSV),--strategy-params-csv "$(STRATEGY_PARAMS_CSV)",)

# Example:
# make train-mtf-regime INSTRUMENTS=XAU_USD,EUR_USD,GBP_USD START=2022-01-01 END=2024-12-31 GPU=auto
train-mtf-regime:
	./.venv/bin/python scripts/train_multiframe_regime_model.py \
		--instruments "$(if $(INSTRUMENTS),$(INSTRUMENTS),XAU_USD,EUR_USD,GBP_USD)" \
		--start "$(if $(START),$(START),2022-01-01)" \
		--end "$(if $(END),$(END),2024-12-31)" \
		--gpu "$(if $(GPU),$(GPU),auto)"

# Example:
# make mtf-prepare-data INSTRUMENTS=EUR_USD,GBP_USD,USD_JPY,XAU_USD BASE_TF=M15 HTF1=H1 HTF2=H4 PREP_EXTRA_TFS=D1 FULL_START=2015-01-01 OOS_END=2025-12-31 CHUNK_MONTHS=3
mtf-prepare-data:
	./.venv/bin/python scripts/run_mtf_baseline_pipeline.py \
		--mode prepare-data \
		--instruments "$(if $(INSTRUMENTS),$(INSTRUMENTS),EUR_USD,GBP_USD,USD_JPY,XAU_USD)" \
		--base-tf "$(if $(BASE_TF),$(BASE_TF),M15)" \
		--htf-1 "$(if $(HTF1),$(HTF1),H1)" \
		--htf-2 "$(if $(HTF2),$(HTF2),H4)" \
		--prepare-extra-tfs "$(if $(PREP_EXTRA_TFS),$(PREP_EXTRA_TFS),D1)" \
		--full-start "$(if $(FULL_START),$(FULL_START),2015-01-01)" \
		--oos-end "$(if $(OOS_END),$(OOS_END),2025-12-31)" \
		--chunk-months "$(if $(CHUNK_MONTHS),$(CHUNK_MONTHS),3)"

# Example:
# make mtf-train INSTRUMENTS=EUR_USD,GBP_USD,USD_JPY,XAU_USD BASE_TF=M15 HTF1=H1 HTF2=H4 FINE_START=2022-01-01 FINE_END=2024-12-31 FULL_START=2015-01-01 FULL_END=2024-12-31 GPU=on
mtf-train:
	./.venv/bin/python scripts/run_mtf_baseline_pipeline.py \
		--mode train \
		--instruments "$(if $(INSTRUMENTS),$(INSTRUMENTS),EUR_USD,GBP_USD,USD_JPY,XAU_USD)" \
		--base-tf "$(if $(BASE_TF),$(BASE_TF),M15)" \
		--htf-1 "$(if $(HTF1),$(HTF1),H1)" \
		--htf-2 "$(if $(HTF2),$(HTF2),H4)" \
		--fine-start "$(if $(FINE_START),$(FINE_START),2022-01-01)" \
		--fine-end "$(if $(FINE_END),$(FINE_END),2024-12-31)" \
		--full-start "$(if $(FULL_START),$(FULL_START),2015-01-01)" \
		--full-end "$(if $(FULL_END),$(FULL_END),2024-12-31)" \
		--gpu "$(if $(GPU),$(GPU),auto)"

# Example:
# make mtf-eval INSTRUMENTS=EUR_USD,GBP_USD,USD_JPY,XAU_USD EVAL_TFS=H1,M15 OOS_START=2025-01-01 OOS_END=2025-12-31 STRATEGY_PARAMS_CSV=data/research/<shortlist.csv>
mtf-eval:
	./.venv/bin/python scripts/run_mtf_baseline_pipeline.py \
		--mode eval \
		--instruments "$(if $(INSTRUMENTS),$(INSTRUMENTS),EUR_USD,GBP_USD,USD_JPY,XAU_USD)" \
		--eval-tfs "$(if $(EVAL_TFS),$(EVAL_TFS),H1,M15)" \
		--oos-start "$(if $(OOS_START),$(OOS_START),2025-01-01)" \
		--oos-end "$(if $(OOS_END),$(OOS_END),2025-12-31)" \
		--risk-per-trade-pct "$(if $(RISK_PER_TRADE_PCT),$(RISK_PER_TRADE_PCT),0.01)" \
		--max-notional-exposure-pct "$(if $(MAX_NOTIONAL_EXPOSURE_PCT),$(MAX_NOTIONAL_EXPOSURE_PCT),1.0)" \
		--min-quantity "$(if $(MIN_QUANTITY),$(MIN_QUANTITY),1)" \
		--max-quantity "$(if $(MAX_QUANTITY),$(MAX_QUANTITY),100000)" \
		--max-drawdown-stop-pct "$(if $(MAX_DRAWDOWN_STOP_PCT),$(MAX_DRAWDOWN_STOP_PCT),0.20)" \
		--daily-loss-limit-pct "$(if $(DAILY_LOSS_LIMIT_PCT),$(DAILY_LOSS_LIMIT_PCT),0.05)" \
		--financing "$(if $(FINANCING),$(FINANCING),on)" \
		--default-financing-long-rate "$(if $(DEFAULT_FINANCING_LONG_RATE),$(DEFAULT_FINANCING_LONG_RATE),0.03)" \
		--default-financing-short-rate "$(if $(DEFAULT_FINANCING_SHORT_RATE),$(DEFAULT_FINANCING_SHORT_RATE),0.03)" \
		--rollover-hour-utc "$(if $(ROLLOVER_HOUR_UTC),$(ROLLOVER_HOUR_UTC),22)" \
		$(if $(STRATEGY_PARAMS_CSV),--strategy-params-csv "$(STRATEGY_PARAMS_CSV)",) \
		$(if $(MANIFEST_JSON),--manifest-json "$(MANIFEST_JSON)",)

# Example:
# make mtf-pipeline INSTRUMENTS=EUR_USD,GBP_USD,USD_JPY,XAU_USD BASE_TF=M15 HTF1=H1 HTF2=H4 EVAL_TFS=H1,M15 FINE_START=2022-01-01 FINE_END=2024-12-31 FULL_START=2015-01-01 FULL_END=2024-12-31 OOS_START=2025-01-01 OOS_END=2025-12-31 GPU=on
mtf-pipeline:
	./.venv/bin/python scripts/run_mtf_baseline_pipeline.py \
		--mode full \
		--instruments "$(if $(INSTRUMENTS),$(INSTRUMENTS),EUR_USD,GBP_USD,USD_JPY,XAU_USD)" \
		--base-tf "$(if $(BASE_TF),$(BASE_TF),M15)" \
		--htf-1 "$(if $(HTF1),$(HTF1),H1)" \
		--htf-2 "$(if $(HTF2),$(HTF2),H4)" \
		--prepare-extra-tfs "$(if $(PREP_EXTRA_TFS),$(PREP_EXTRA_TFS),D1)" \
		--eval-tfs "$(if $(EVAL_TFS),$(EVAL_TFS),H1,M15)" \
		--fine-start "$(if $(FINE_START),$(FINE_START),2022-01-01)" \
		--fine-end "$(if $(FINE_END),$(FINE_END),2024-12-31)" \
		--full-start "$(if $(FULL_START),$(FULL_START),2015-01-01)" \
		--full-end "$(if $(FULL_END),$(FULL_END),2024-12-31)" \
		--oos-start "$(if $(OOS_START),$(OOS_START),2025-01-01)" \
		--oos-end "$(if $(OOS_END),$(OOS_END),2025-12-31)" \
		--gpu "$(if $(GPU),$(GPU),auto)" \
		--chunk-months "$(if $(CHUNK_MONTHS),$(CHUNK_MONTHS),3)" \
		--risk-per-trade-pct "$(if $(RISK_PER_TRADE_PCT),$(RISK_PER_TRADE_PCT),0.01)" \
		--max-notional-exposure-pct "$(if $(MAX_NOTIONAL_EXPOSURE_PCT),$(MAX_NOTIONAL_EXPOSURE_PCT),1.0)" \
		--min-quantity "$(if $(MIN_QUANTITY),$(MIN_QUANTITY),1)" \
		--max-quantity "$(if $(MAX_QUANTITY),$(MAX_QUANTITY),100000)" \
		--max-drawdown-stop-pct "$(if $(MAX_DRAWDOWN_STOP_PCT),$(MAX_DRAWDOWN_STOP_PCT),0.20)" \
		--daily-loss-limit-pct "$(if $(DAILY_LOSS_LIMIT_PCT),$(DAILY_LOSS_LIMIT_PCT),0.05)" \
		--financing "$(if $(FINANCING),$(FINANCING),on)" \
		--default-financing-long-rate "$(if $(DEFAULT_FINANCING_LONG_RATE),$(DEFAULT_FINANCING_LONG_RATE),0.03)" \
		--default-financing-short-rate "$(if $(DEFAULT_FINANCING_SHORT_RATE),$(DEFAULT_FINANCING_SHORT_RATE),0.03)" \
		--rollover-hour-utc "$(if $(ROLLOVER_HOUR_UTC),$(ROLLOVER_HOUR_UTC),22)" \
		$(if $(STRATEGY_PARAMS_CSV),--strategy-params-csv "$(STRATEGY_PARAMS_CSV)",)

# Example:
# make mtf-clean OLDER_THAN_DAYS=7 DELETE_AFTER_ARCHIVE=1
mtf-clean:
	./.venv/bin/python scripts/cleanup_research_artifacts.py \
		--output-dir "$(if $(OUTPUT_DIR),$(OUTPUT_DIR),data/research)" \
		--archive-dir "$(if $(ARCHIVE_DIR),$(ARCHIVE_DIR),data/research/archive)" \
		--older-than-days "$(if $(OLDER_THAN_DAYS),$(OLDER_THAN_DAYS),7)" \
		$(if $(DELETE_AFTER_ARCHIVE),--delete-after-archive,) \
		$(if $(DRY_RUN),--dry-run,)

# Example:
# make xau-mtf-dev FINE_START=2022-01-01 FINE_END=2024-12-31 FULL_START=2015-01-01 FULL_END=2024-12-31 OOS_START=2025-01-01 OOS_END=2025-12-31 GPU=on
xau-mtf-dev:
	./.venv/bin/python scripts/run_mtf_baseline_pipeline.py \
		--mode full \
		--instruments "XAU_USD" \
		--base-tf "M15" \
		--htf-1 "H1" \
		--htf-2 "H4" \
		--prepare-extra-tfs "D1" \
		--eval-tfs "H1,M15" \
		--fine-start "$(if $(FINE_START),$(FINE_START),2022-01-01)" \
		--fine-end "$(if $(FINE_END),$(FINE_END),2024-12-31)" \
		--full-start "$(if $(FULL_START),$(FULL_START),2015-01-01)" \
		--full-end "$(if $(FULL_END),$(FULL_END),2024-12-31)" \
		--oos-start "$(if $(OOS_START),$(OOS_START),2025-01-01)" \
		--oos-end "$(if $(OOS_END),$(OOS_END),2025-12-31)" \
		--gpu "$(if $(GPU),$(GPU),auto)" \
		--chunk-months "$(if $(CHUNK_MONTHS),$(CHUNK_MONTHS),3)" \
		--risk-per-trade-pct "$(if $(RISK_PER_TRADE_PCT),$(RISK_PER_TRADE_PCT),0.01)" \
		--max-notional-exposure-pct "$(if $(MAX_NOTIONAL_EXPOSURE_PCT),$(MAX_NOTIONAL_EXPOSURE_PCT),1.0)" \
		--min-quantity "$(if $(MIN_QUANTITY),$(MIN_QUANTITY),1)" \
		--max-quantity "$(if $(MAX_QUANTITY),$(MAX_QUANTITY),100000)" \
		--max-drawdown-stop-pct "$(if $(MAX_DRAWDOWN_STOP_PCT),$(MAX_DRAWDOWN_STOP_PCT),0.20)" \
		--daily-loss-limit-pct "$(if $(DAILY_LOSS_LIMIT_PCT),$(DAILY_LOSS_LIMIT_PCT),0.05)" \
		--financing "$(if $(FINANCING),$(FINANCING),on)" \
		--default-financing-long-rate "$(if $(DEFAULT_FINANCING_LONG_RATE),$(DEFAULT_FINANCING_LONG_RATE),0.03)" \
		--default-financing-short-rate "$(if $(DEFAULT_FINANCING_SHORT_RATE),$(DEFAULT_FINANCING_SHORT_RATE),0.03)" \
		--rollover-hour-utc "$(if $(ROLLOVER_HOUR_UTC),$(ROLLOVER_HOUR_UTC),22)" \
		$(if $(STRATEGY_PARAMS_CSV),--strategy-params-csv "$(STRATEGY_PARAMS_CSV)",)

# Example:
# make xau-multi-session-train CONFIG=config/xau_multi_session_pipeline.default.json
xau-multi-session-train:
	./.venv/bin/python scripts/xau_multi_session_pipeline.py \
		--config "$(if $(CONFIG),$(CONFIG),config/xau_multi_session_pipeline.default.json)"

# Example:
# make xau-multi-session-smoke CONFIG=config/xau_multi_session_pipeline.smoke.json REGIME_TRAIN_YEARS=3
xau-multi-session-smoke:
	REGIME_DISABLE_DOWNLOAD=1 REGIME_TRAIN_YEARS="$(if $(REGIME_TRAIN_YEARS),$(REGIME_TRAIN_YEARS),3)" ./.venv/bin/python scripts/xau_multi_session_pipeline.py \
		--config "$(if $(CONFIG),$(CONFIG),config/xau_multi_session_pipeline.smoke.json)"

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

# Example:
# DISCORD_BOT_TOKEN=... DISCORD_CHANNEL_ID=... make discord-operator-bot ALERT_MIN_SEVERITY=warning
discord-operator-bot:
	./.venv/bin/python scripts/discord_operator_bot.py \
		--poll-seconds "$(if $(POLL_SECONDS),$(POLL_SECONDS),3)" \
		--alert-min-severity "$(if $(ALERT_MIN_SEVERITY),$(ALERT_MIN_SEVERITY),warning)" \
		--commands-prefix "$(if $(COMMANDS_PREFIX),$(COMMANDS_PREFIX),!)"

# Example:
# DISCORD_EXEC_BOT_TOKEN=... DISCORD_EXEC_CHANNEL_ID=1477609642258337954 make discord-execution-notifier
discord-execution-notifier:
	./.venv/bin/python scripts/discord_execution_notifier.py \
		--balance-timeout-seconds "$(if $(BALANCE_TIMEOUT_SECONDS),$(BALANCE_TIMEOUT_SECONDS),10)"

# Example:
# REGIME_MODEL_JSON=models/active/multiframe_regime_model_20260228_194306.json make trading-supervisor
trading-supervisor:
	./.venv/bin/python scripts/trading_supervisor.py \
		--project-root "$(if $(PROJECT_ROOT),$(PROJECT_ROOT),/home/joe/Desktop/Algo_trading/oanda-trading-system)" \
		--poll-seconds "$(if $(POLL_SECONDS),$(POLL_SECONDS),2)"

# Example:
# make trading-journal-agent OUTPUT_DIR=data/reports/trading_journal
trading-journal-agent:
	./.venv/bin/python scripts/trading_journal_agent.py \
		--output-dir "$(if $(OUTPUT_DIR),$(OUTPUT_DIR),data/reports/trading_journal)"

# Example:
# make systemd-user-install PROJECT_ROOT=/home/joe/Desktop/Algo_trading/oanda-trading-system
systemd-user-install:
	./scripts/install_systemd_user_services.sh "$(if $(PROJECT_ROOT),$(PROJECT_ROOT),/home/joe/Desktop/Algo_trading/oanda-trading-system)"

systemd-user-enable:
	systemctl --user enable --now oanda-trading-supervisor.service
	systemctl --user enable --now oanda-discord-operator.service
	systemctl --user enable --now oanda-discord-execution-notifier.service

systemd-user-enable-infra:
	systemctl --user enable --now oanda-infra.service

systemd-user-status:
	systemctl --user status oanda-infra.service oanda-trading-supervisor.service oanda-discord-operator.service oanda-discord-execution-notifier.service
