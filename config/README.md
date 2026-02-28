# Config Module

Main configuration files:
1. `system.yaml`: Redis/Influx/Prometheus/logging infrastructure.
2. `oanda.yaml`: account credentials, environment, instruments, API behavior.
3. `risk_limits.yaml`: risk controls and circuit-breaker thresholds.
4. `strategies.yaml`: enabled strategy models and parameters.
5. `monitoring.yaml`: health/alerting/metrics settings.
6. `backtesting.yaml`: backtest defaults and execution cost assumptions.

Use environment variables for secrets. Never commit production credentials.

