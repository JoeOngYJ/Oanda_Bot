# AGENTS.md

## Purpose
This document defines:
1. Runtime trading-agent orchestration for this OANDA system.
2. Developer/AI-agent collaboration rules for changes to this repository.

It is the operational source of truth for agent responsibilities, handoffs, and safety controls.

## Scope
This file covers:
1. Runtime agents: `market_data`, `strategy`, `risk`, `execution`, `monitoring`.
2. Developer agents: planning, implementation, and review roles for repo changes.

## Runtime Topology
Primary data path:
1. `Market Data Agent` -> `stream:market_data`
2. `Strategy Agent` -> `stream:signals`
3. `Risk Agent` -> `stream:risk_checks` and `stream:orders`
4. `Execution Agent` -> OANDA order API and `stream:executions`
5. `Monitoring Agent` -> `stream:alerts`, Prometheus metrics

Infrastructure:
1. Redis Streams for inter-agent messaging.
2. InfluxDB for market tick persistence.
3. Prometheus and Grafana for observability.

## Stream Contracts
Contracts are defined by `shared/models.py` and `config/system.yaml`.

| Stream | Producer | Consumer(s) | Model Contract |
|---|---|---|---|
| `stream:market_data` | Market Data Agent | Strategy Agent, Risk Agent | `MarketTick` |
| `stream:signals` | Strategy Agent, Risk Agent (close signals) | Risk Agent | `TradeSignal` |
| `stream:risk_checks` | Risk Agent | Execution Agent, Monitoring Agent | `RiskCheckResult` |
| `stream:orders` | Risk Agent | Execution Agent (future direct order path) | `Order` |
| `stream:executions` | Execution Agent | Risk Agent, Monitoring Agent | `Execution` |
| `stream:alerts` | All agents | Monitoring Agent | Alert payload dictionary |

### Contract Rules
1. Producers must publish JSON-serializable payloads only.
2. Consumers must validate payloads with Pydantic model parsing before business logic.
3. Message changes are breaking unless backward-compatible fields are added only.
4. Stream names must remain synchronized with `config/system.yaml`.

## Runtime Agent Roles

### Market Data Agent
Responsibilities:
1. Subscribe to OANDA pricing stream for configured instruments.
2. Normalize raw ticks into `MarketTick`.
3. Validate data quality and anomaly thresholds.
4. Persist valid ticks to InfluxDB.
5. Publish valid ticks to `stream:market_data`.

Inputs:
1. OANDA pricing stream.
2. `config/oanda.yaml`, `config/system.yaml`.

Outputs:
1. `stream:market_data`.
2. `stream:alerts` for validation anomalies.
3. InfluxDB measurement writes.

Failure behavior:
1. Retry OANDA stream with configured backoff.
2. Continue pipeline on storage errors (non-blocking persistence failures).

### Strategy Agent
Responsibilities:
1. Consume `stream:market_data`.
2. Maintain strategy state (indicator windows/history).
3. Generate `TradeSignal` events.
4. Publish signals to `stream:signals`.

Inputs:
1. `stream:market_data`.
2. `config/strategies.yaml`.

Outputs:
1. `stream:signals`.

Failure behavior:
1. Log and continue processing next tick.
2. Disable only broken strategy implementations if isolated.

### Risk Agent
Responsibilities:
1. Consume `stream:signals`.
2. Run pre-trade checks (size, leverage, stop-loss, exposure, circuit-breaker).
3. Publish risk decisions to `stream:risk_checks`.
4. Publish approved orders to `stream:orders`.
5. Monitor open positions against market ticks.
6. Update risk state from execution reports.

Inputs:
1. `stream:signals`.
2. `stream:market_data`.
3. `stream:executions`.
4. `config/risk_limits.yaml`.

Outputs:
1. `stream:risk_checks`.
2. `stream:orders`.
3. `stream:signals` (risk-generated close signals).

Failure behavior:
1. Default reject posture for malformed/invalid signals.
2. Keep circuit-breaker state conservative when state is uncertain.

### Execution Agent
Responsibilities:
1. Consume approved risk decisions.
2. Construct and submit market orders to OANDA.
3. Track order lifecycle and fills.
4. Publish `Execution` events.
5. Emit alerts for failures and retries.

Inputs:
1. `stream:risk_checks`.
2. OANDA trading REST API.

Outputs:
1. `stream:executions`.
2. `stream:alerts`.

Failure behavior:
1. Retry transient network failures with exponential backoff.
2. Mark orders failed on non-retryable errors.
3. Never silently swallow rejections from broker/API.

### Monitoring Agent
Responsibilities:
1. Collect health/performance metrics.
2. Expose Prometheus metrics endpoint.
3. Process alert stream and route notifications.
4. Monitor stream lag/backlog and raise warnings.

Inputs:
1. `stream:alerts`.
2. Redis stream stats.
3. System metrics.
4. `config/monitoring.yaml`.

Outputs:
1. Prometheus metrics endpoint.
2. Alert notifications (log/email based on config).

Failure behavior:
1. Monitoring failures must not block trading agents.
2. Continue degraded monitoring with local logging if channels fail.

## Startup and Shutdown Runbook

### Startup Order
1. Start infra: `docker-compose up -d`.
2. Verify infra health:
   1. `docker-compose ps`
   2. `redis-cli ping`
   3. `curl http://localhost:8086/health`
3. Start agents in this order:
   1. `python -m agents.market_data.agent`
   2. `python -m agents.monitoring.agent`
   3. `python -m agents.strategy.agent`
   4. `python -m agents.risk.agent`
   5. `python -m agents.execution.agent`
4. Verify readiness:
   1. `redis-cli XLEN stream:market_data`
   2. `redis-cli XLEN stream:signals`
   3. `curl http://localhost:8000/metrics`

### Shutdown Order
1. Stop `execution` first.
2. Stop `strategy` and `risk`.
3. Stop `market_data`.
4. Stop `monitoring`.
5. Stop infra if required: `docker-compose down`.

### Emergency Stop
1. Stop execution agent immediately.
2. Stop strategy and risk agents.
3. Keep monitoring online for diagnostics.
4. Capture logs and stream lengths before restart.

## Safety and Approval Policy
Mixed enforcement model:
1. `practice` environment:
   1. Autonomous operation is allowed under configured risk limits.
2. `live` environment:
   1. Human approval is required before enabling order routing.
   2. Human approval is required before switching `oanda.environment` to `live`.
   3. Human approval is required for risk limit loosening or strategy parameter widening.

Mandatory controls for `live`:
1. Two-person review for:
   1. `config/oanda.yaml`
   2. `config/risk_limits.yaml`
   3. `agents/execution/*`
   4. `agents/risk/*`
2. Pre-deploy checks:
   1. Unit/integration tests pass.
   2. Dry-run in `practice` with monitoring green.
3. Circuit-breaker must remain enabled.

## Change Control
Any change to stream contracts, models, risk rules, or execution logic must include:
1. Contract impact note.
2. Updated tests for changed behavior.
3. Runbook updates if startup/failure behavior changes.
4. Clear rollback plan.

## Developer/AI Collaboration Agents

### Planner Agent
Responsibilities:
1. Convert requests into decision-complete implementation plans.
2. Identify safety implications and test requirements.

Output artifact:
1. A concrete implementation plan with acceptance criteria.

### Implementer Agent
Responsibilities:
1. Execute the approved plan with minimal scope drift.
2. Keep changes atomic and traceable.
3. Run appropriate tests and checks.

Output artifact:
1. Code/config/docs changes plus verification summary.

### Reviewer Agent
Responsibilities:
1. Prioritize correctness, risk, and regression detection.
2. Validate message contracts, safety gates, and test coverage.
3. Reject changes that weaken live-trading controls without explicit approval.

Output artifact:
1. Severity-ordered findings and required fixes.

## Ownership and Escalation
Assign and maintain owners for:
1. `market_data`
2. `strategy`
3. `risk`
4. `execution`
5. `monitoring`
6. Infrastructure (`redis`, `influxdb`, observability stack)

Escalation priority:
1. Safety-impacting incidents (possible bad orders).
2. Execution outages.
3. Data integrity issues.
4. Monitoring degradation.

## References
1. `docs/ARCHITECTURE.md`
2. `docs/OPERATIONS.md`
3. `config/system.yaml`
4. `config/oanda.yaml`
5. `config/risk_limits.yaml`
6. `shared/models.py`

