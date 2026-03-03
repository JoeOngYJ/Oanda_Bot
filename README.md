# OANDA Trading System

Multi-agent FX trading and backtesting platform built around OANDA v20 APIs.

## What This Repo Contains
1. Live-style agent pipeline:
   1. `market_data` -> `strategy` -> `risk` -> `execution` -> `monitoring`
2. Backtesting stack:
   1. Data download/warehouse
   2. Strategy simulation
   3. Execution modeling with spread + commission
3. Shared infrastructure:
   1. Pydantic models
   2. Redis Streams message bus
   3. YAML config loading + env substitution
4. Operational support:
   1. Docker Compose for Redis/InfluxDB/Grafana/Prometheus
   2. Test suite across unit, integration, performance, stress, failover

## Quick Start

### 1. Install
```bash
cd /home/joe/Desktop/Algo_trading/oanda-trading-system
python -m pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
```

Set required secrets in `.env`:
1. `OANDA_ACCOUNT_ID`
2. `OANDA_API_TOKEN`
3. `INFLUXDB_TOKEN`

### 3. Start Infrastructure
```bash
docker-compose up -d
docker-compose ps
```

### 4. Run Tests
```bash
pytest -q tests/backtesting
pytest -q tests/test_execution tests/test_risk tests/test_market_data
```

### 5. Run Agents
Use:
```bash
bash scripts/run_all_agents.sh
```
Then run each printed command in its own terminal.

## Documentation Hub
0. Navigation guide: [docs/reference/NAVIGATION.md](/home/joe/Desktop/Algo_trading/oanda-trading-system/docs/reference/NAVIGATION.md)
1. Model navigation guide: [docs/reference/MODEL_NAVIGATION.md](/home/joe/Desktop/Algo_trading/oanda-trading-system/docs/reference/MODEL_NAVIGATION.md)
2. Architecture: [docs/ARCHITECTURE.md](/home/joe/Desktop/Algo_trading/oanda-trading-system/docs/ARCHITECTURE.md)
2. Operations: [docs/OPERATIONS.md](/home/joe/Desktop/Algo_trading/oanda-trading-system/docs/OPERATIONS.md)
3. Deployment: [docs/DEPLOYMENT.md](/home/joe/Desktop/Algo_trading/oanda-trading-system/docs/DEPLOYMENT.md)
4. Troubleshooting: [docs/TROUBLESHOOTING.md](/home/joe/Desktop/Algo_trading/oanda-trading-system/docs/TROUBLESHOOTING.md)
5. Feature catalog: [docs/FEATURE_CATALOG.md](/home/joe/Desktop/Algo_trading/oanda-trading-system/docs/FEATURE_CATALOG.md)
6. Repo map and cleanup guide: [docs/REPO_STRUCTURE.md](/home/joe/Desktop/Algo_trading/oanda-trading-system/docs/REPO_STRUCTURE.md)
7. Professionalization roadmap: [docs/PROFESSIONALIZATION_PLAN.md](/home/joe/Desktop/Algo_trading/oanda-trading-system/docs/PROFESSIONALIZATION_PLAN.md)
8. HTML project overview: [docs/overview.html](/home/joe/Desktop/Algo_trading/oanda-trading-system/docs/overview.html)
9. Src migration status: [docs/MIGRATION_STATUS.md](/home/joe/Desktop/Algo_trading/oanda-trading-system/docs/MIGRATION_STATUS.md)
10. Profitability playbook: [docs/PROFITABILITY_PLAYBOOK.md](/home/joe/Desktop/Algo_trading/oanda-trading-system/docs/PROFITABILITY_PLAYBOOK.md)
11. Strategy scorecard template: [docs/STRATEGY_SCORECARD_TEMPLATE.md](/home/joe/Desktop/Algo_trading/oanda-trading-system/docs/STRATEGY_SCORECARD_TEMPLATE.md)
12. Strategy scorecard CSV: [strategy_scorecard_template.csv](/home/joe/Desktop/Algo_trading/oanda-trading-system/data/templates/strategy_scorecard_template.csv)
13. Scorecard generator script: `python scripts/scorecard_new_week.py --strategy <name> --environment <paper|live_micro|live_scaled|research>`
14. Strategy research runner: `python scripts/run_strategy_research.py --instrument EUR_USD --tf H1 --start 2023-01-01 --end 2024-01-01` (or add `--demo-bars 2000`)
15. Universe research runner: `python scripts/run_universe_research.py --instruments EUR_USD,GBP_USD,USD_JPY,XAU_USD --base-tf M15 --wf-windows 8 --min-stability 0.30 --min-trades 2 --max-corr 0.70`
16. Strategy focus plan: [docs/STRATEGY_FOCUS_PLAN.md](/home/joe/Desktop/Algo_trading/oanda-trading-system/docs/STRATEGY_FOCUS_PLAN.md)
17. Regime GPU research: `python scripts/run_regime_gpu_research.py --instrument EUR_USD --tf M15 --regimes 4 --gpu auto --demo-bars 3000`
18. Strategy evidence map: [docs/STRATEGY_EVIDENCE.md](/home/joe/Desktop/Algo_trading/oanda-trading-system/docs/STRATEGY_EVIDENCE.md)
19. Intermarket MTF strategy (cross-pair + multi-timeframe) runs via universe research and appears as `IntermarketMTFConfluence`.
20. GPU strategy pre-screener:
   1. Run fast pre-screen: `python scripts/run_gpu_prescreener.py --instrument EUR_USD --tf M15 --start 2022-01-01 --end 2025-12-31 --gpu auto --top-n 20`
   2. Use shortlist in full walk-forward: `python scripts/run_universe_research.py --instruments EUR_USD,GBP_USD,USD_JPY,XAU_USD --base-tf M15 --candidate-shortlist data/research/<gpu_shortlist.csv>`
   3. One-command pipeline: `make gpu-universe-pipeline INSTRUMENT=EUR_USD INSTRUMENTS=EUR_USD,GBP_USD,USD_JPY,XAU_USD TF=M15 BASE_TF=M15 START=2022-01-01 END=2025-12-31 GPU=auto TOP_N=20 WF_WINDOWS=8 MIN_STABILITY=0.30 MIN_TRADES=2 MAX_CORR=0.70`
21. Real-time style backtester pipeline:
   1. `python scripts/run_realtime_backtest.py --instrument EUR_USD --tf M15 --start 2024-01-01 --end 2024-12-31 --fill-mode next_open`
   2. Make target: `make realtime-backtest INSTRUMENT=EUR_USD TF=M15 FILL_MODE=next_open`
   3. Optional state snapshots:
      1. `--state-snapshot-path data/research/state_snapshots.jsonl --snapshot-every-bars 1`
22. Runtime regime classifier + auto strategy selection:
   1. Generate runtime model: `python scripts/run_regime_gpu_research.py --instrument EUR_USD --tf M15 --gpu auto`
   2. Run regime strategy selection (default: ensemble decision): `python scripts/run_regime_runtime_backtest.py --model-json data/research/<...>_runtime_model.json --instrument EUR_USD --tf M15 --start 2025-01-01 --end 2025-12-31 --fill-mode next_open --decision-mode ensemble`
   3. Router fallback mode: `--decision-mode router`
23. Working strategies report: [docs/WORKING_STRATEGIES.md](/home/joe/Desktop/Algo_trading/oanda-trading-system/docs/WORKING_STRATEGIES.md)
24. XAU breakout-only strategy spec: [docs/XAU_BREAKOUT_STRATEGY.md](/home/joe/Desktop/Algo_trading/oanda-trading-system/docs/XAU_BREAKOUT_STRATEGY.md)
25. Execution control commands:
   1. `make exec-kill-on` / `make exec-kill-off`
   2. `make exec-shadow-on` / `make exec-shadow-off`
26. Live execution guardrail:
   1. In `live` env, broker execution is blocked unless `EXECUTION_LIVE_ENABLED=true`.
   2. Use shadow mode for safe dry runs.
27. Operator status API:
   1. Start: `make exec-status-api PORT=8010`
   2. Health: `GET /health`
   3. Safety state: `GET /execution/state`
28. Multi-timeframe baseline framework: [docs/MTF_BASELINE.md](/home/joe/Desktop/Algo_trading/oanda-trading-system/docs/MTF_BASELINE.md)
29. Multi-timeframe model architecture (XAU-first): [docs/MTF_MODEL_ARCHITECTURE.md](/home/joe/Desktop/Algo_trading/oanda-trading-system/docs/MTF_MODEL_ARCHITECTURE.md)
30. MTF baseline pipeline targets:
   1. `make mtf-prepare-data`
   2. `make mtf-train`
   3. `make mtf-eval`
   4. `make mtf-pipeline`
   5. `make xau-mtf-dev`
31. Intermediate artifact retention:
   1. `make mtf-clean OLDER_THAN_DAYS=7 DELETE_AFTER_ARCHIVE=1`
32. Systemd + VPS runbook: [docs/SYSTEMD_VPS_RUNBOOK.md](/home/joe/Desktop/Algo_trading/oanda-trading-system/docs/SYSTEMD_VPS_RUNBOOK.md)
33. Download fix note (archived): [docs/notes/DOWNLOAD_FIX_SUMMARY.md](/home/joe/Desktop/Algo_trading/oanda-trading-system/docs/notes/DOWNLOAD_FIX_SUMMARY.md)
34. XAU multi-session deterministic pipeline:
   1. Full run: `make xau-multi-session-train CONFIG=config/xau_multi_session_pipeline.default.json`
   2. Smoke run: `make xau-multi-session-smoke CONFIG=config/xau_multi_session_pipeline.smoke.json REGIME_TRAIN_YEARS=3`
35. XAU multi-session configs:
   1. Default: [config/xau_multi_session_pipeline.default.json](/home/joe/Desktop/Algo_trading/oanda-trading-system/config/xau_multi_session_pipeline.default.json)
   2. Smoke: [config/xau_multi_session_pipeline.smoke.json](/home/joe/Desktop/Algo_trading/oanda-trading-system/config/xau_multi_session_pipeline.smoke.json)

## Core Streams
Configured in `config/system.yaml`:
1. `stream:market_data`
2. `stream:signals`
3. `stream:risk_checks`
4. `stream:orders`
5. `stream:executions`
6. `stream:alerts`

## Safety Notes
1. `config/oanda.yaml` currently supports `practice` and `live`.
2. Use `practice` for development and validation.
3. Do not route to `live` without explicit approval and full test checks.
4. Enforce controls in [AGENTS.md](/home/joe/Desktop/Algo_trading/oanda-trading-system/AGENTS.md).

## Development Conventions
1. Keep message schemas aligned with `shared/models.py`.
2. Update tests when changing risk or execution behavior.
3. Keep docs current for any stream/contract change.
4. Prefer non-breaking changes unless a migration plan is documented.

## License
Proprietary internal project.
