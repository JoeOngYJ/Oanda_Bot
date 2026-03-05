# Feature Catalog

This file lists current capabilities, grouped by domain.

## 1. Live Agent Pipeline

### 1.1 Market Data
1. OANDA streaming client integration.
2. Tick normalization to shared `MarketTick` schema.
3. Tick validation rules (spread sanity, price jumps, staleness checks).
4. InfluxDB persistence for ticks.
5. Redis publication to `stream:market_data`.

### 1.2 Strategy
1. Multi-strategy loading from config.
2. Implemented models:
   1. Moving Average Crossover
   2. RSI Mean Reversion
3. Signal generation with confidence + rationale metadata.
4. Redis publication to `stream:signals`.

### 1.3 Risk
1. Pre-trade checks:
   1. Max order size
   2. Max position size
   3. Stop-loss required and distance checks
   4. Leverage checks
   5. Account balance checks
   6. Total exposure checks
   7. Daily loss checks
   8. Circuit breaker checks
2. Position monitoring and risk-triggered close signal generation.
3. Circuit breaker with loss velocity and drawdown controls.

### 1.4 Execution
1. OANDA market order payload builder.
2. Retry logic for transient failures.
3. Fill parsing into shared `Execution` model.
4. Execution alerts to `stream:alerts`.
5. Idempotency key attached to orders (`sig:<signal_id>`), including broker client extension.
6. Durable duplicate signal suppression via Redis (`execution:signal:*`) to avoid re-executing across restarts.
7. Shadow mode to simulate fills without broker submission.
8. Kill-switch support to hard-block order execution.
9. Execution control stream (`stream:execution_control`) for operator commands.
10. Live-execution guardrail: live broker execution is blocked unless explicitly enabled.

### 1.5 Monitoring
1. Health checks and metric collection.
2. Prometheus endpoint serving.
3. Alert stream processing.
4. Stream lag monitoring.
5. Execution safety state metrics:
   1. `execution_kill_switch_active`
   2. `execution_shadow_mode_active`
6. Operator JSON endpoint (`scripts/execution_status_api.py`):
   1. `GET /execution/state`
   2. Returns kill-switch, shadow mode, live guardrail, and idempotency cache counters.

## 2. Backtesting

### 2.1 Data Layer
1. OANDA historical downloader.
2. Warehouse storage with Parquet and CSV fallback.
3. Data manager for cache + non-destructive range extension.
4. Timeframe resampling support.

### 2.2 Strategy Simulation
1. `StrategyBase` abstraction.
2. Example multi-timeframe trend strategy.
3. Signal model with direction, quantity, confidence, and price levels.
4. Real-time style pipeline backtester (`backtesting/core/backtester.py`) with explicit stage hooks:
   1. Feature engineering
   2. Regime prediction
   3. Risk assessment before execution
5. Central dynamic `SystemState` (`backtesting/core/state.py`) tracks:
   1. Positions by symbol
   2. Available cash and marked-to-market equity
   3. Historical PnL / equity curve
   4. Last detected regime
   5. In-memory market and feature buffers
6. Optional state snapshot persistence to JSONL (`state.snapshot_path`) for recovery/analysis.

### 2.3 Execution Modeling
1. Mid-to-fill conversion with spread + slippage model.
2. Configurable fill mode:
   1. `touch` (bar high/low touch logic)
   2. `next_open` (signal at bar close, fill at next bar open)
2. OANDA-style typical spread map defaults (overridable).
3. Commission model variants:
   1. Spread-only pricing
   2. OANDA core per-10k-unit pricing
   3. Flat per-trade compatibility mode
4. Entry and SL/TP exit fill simulation.
5. Per-fill commission accounting.

### 2.4 Metrics and Results
1. Backtest run returns:
   1. Total trades
   2. Win rate
   3. Sharpe estimate
   4. Max drawdown
   5. Final equity
   6. Total fees
   7. Trade list and equity curve

### 2.5 Research Acceleration
1. Cost-aware proxy scoring includes spread, slippage, and OANDA core commission approximation.
2. Strategy shortlist output can be fed into universe walk-forward runner using:
   1. `--candidate-shortlist <csv>`

### 2.6 Regime Runtime Selection
1. Regime model export from GPU research includes:
   1. feature columns
   2. train mean/std
   3. centroid matrix
   4. regime-to-strategy mapping
2. Runtime classifier:
   1. `RegimeFeatureEngineer` computes streaming regime features.
   2. `KMeansRegimePredictor` assigns nearest centroid regime per bar and outputs regime probabilities (softmax on centroid distance).
3. Strategy auto-selection:
   1. `RegimeSwitchRouter` routes strategy logic by current regime.
   2. `RegimeEnsembleDecisionStrategy` runs multiple strategy modules in parallel and applies regime-aware weighted voting.

## 3. Config and Infrastructure
1. YAML config loading with env var substitution.
2. Redis Streams messaging abstraction with consumer groups.
3. Docker Compose stack:
   1. Redis
   2. InfluxDB
   3. Grafana
   4. Prometheus

## 4. Testing Coverage
1. Unit tests for shared models/config/message bus.
2. Agent-specific tests:
   1. Market data
   2. Strategy
   3. Risk
   4. Execution
   5. Monitoring
3. Backtesting tests:
   1. Engine
   2. Data manager
   3. Strategy import
   4. Execution costs and SL/TP behavior
4. Integration tests:
   1. End-to-end flow
   2. OANDA/infrastructure integrations
5. Performance, stress, and failover tests.

## 5. Current Gaps (Known)
1. Backtesting analytics modules (`metrics.py`, `reports.py`) are still basic/stub-level.
2. Some live-agent data contracts still need full harmonization across risk/execution.
3. Production hardening items (secrets lifecycle, release automation, migration discipline) are in progress.
