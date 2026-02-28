# SKILLS.md

## Purpose
This document defines reusable project skills for operating and evolving this multi-agent OANDA trading system.

Each skill card includes:
1. `skill_id`
2. Purpose
3. Triggers
4. Inputs
5. Outputs
6. Dependencies
7. Failure modes
8. Runbook references

## Skill Invocation Model
1. Select the smallest skill set that completes the task.
2. Prefer sequential composition when outputs from one skill become required inputs for the next.
3. For `live`-impacting operations, apply approval gates from `AGENTS.md`.

## Skill Catalog

### Skill: `market-data-ops`
Purpose:
1. Operate, diagnose, and recover market data ingestion.

Triggers:
1. "market data is stale"
2. "ticks are not flowing"
3. "validate OANDA stream health"

Inputs:
1. OANDA config (`config/oanda.yaml`)
2. Stream stats (`stream:market_data`)
3. Market data logs

Outputs:
1. Health diagnosis summary
2. Recovery actions executed or proposed
3. Post-recovery validation evidence

Dependencies:
1. Redis, OANDA connectivity, InfluxDB
2. `agents/market_data/*`

Failure modes:
1. OANDA stream disconnect loops
2. Bad tick normalization/validation causing drop storms
3. Influx write failures

Runbook refs:
1. `docs/OPERATIONS.md`
2. `AGENTS.md` (Market Data Agent + startup checks)

### Skill: `strategy-validation`
Purpose:
1. Validate strategy signal quality and strategy-agent behavior.

Triggers:
1. "signals stopped"
2. "too many/too few signals"
3. "validate MA/RSI strategy behavior"

Inputs:
1. `config/strategies.yaml`
2. Strategy logs
3. Recent `stream:market_data` and `stream:signals` samples

Outputs:
1. Signal generation diagnosis
2. Parameter or logic change recommendations
3. Test scenarios to protect against recurrence

Dependencies:
1. `agents/strategy/*`
2. Shared model contracts (`TradeSignal`, `MarketTick`)

Failure modes:
1. Strategy class/config mismatch
2. Indicator history not warming up
3. Invalid payload parsing from stream messages

Runbook refs:
1. `AGENTS.md` (Strategy Agent)
2. `docs/ARCHITECTURE.md`

### Skill: `risk-limit-audit`
Purpose:
1. Audit risk enforcement paths and validate limit behavior.

Triggers:
1. "risk agent allowed bad signal"
2. "circuit breaker behavior check"
3. "review risk config changes"

Inputs:
1. `config/risk_limits.yaml`
2. Risk/rejection logs
3. Samples from `stream:signals`, `stream:risk_checks`, `stream:executions`

Outputs:
1. Risk control audit report
2. List of violated or untested controls
3. Required fixes and tests

Dependencies:
1. `agents/risk/*`
2. `shared/models.py`

Failure modes:
1. Side/value interpretation errors
2. Stale risk state
3. Circuit-breaker not activating when expected

Runbook refs:
1. `AGENTS.md` (Risk Agent + safety policy)
2. `docs/OPERATIONS.md`

### Skill: `execution-reliability`
Purpose:
1. Improve and validate order submission, retries, and execution reporting.

Triggers:
1. "orders failing"
2. "execution retries are noisy"
3. "broker rejects are increasing"

Inputs:
1. Execution logs and alerts
2. `config/oanda.yaml`
3. `stream:risk_checks`, `stream:executions`

Outputs:
1. Root-cause analysis for execution failures
2. Retry/error handling improvements
3. Verification results for success/failure paths

Dependencies:
1. `agents/execution/*`
2. OANDA API client behavior

Failure modes:
1. Non-retryable errors treated as retryable
2. Model/field mismatches between risk and execution
3. Missing execution publication after fill

Runbook refs:
1. `AGENTS.md` (Execution Agent + live gating)
2. `docs/TROUBLESHOOTING.md`

### Skill: `monitoring-incident-response`
Purpose:
1. Triage, contain, and resolve production incidents across agent boundaries.

Triggers:
1. "critical alert fired"
2. "system unstable"
3. "stream backlog spike"

Inputs:
1. Alert stream and monitoring logs
2. Prometheus metrics endpoint
3. Stream lag/backlog stats

Outputs:
1. Incident severity classification
2. Containment steps
3. Recovery and post-incident action list

Dependencies:
1. `agents/monitoring/*`
2. Prometheus/Grafana stack

Failure modes:
1. Alert fatigue from noisy thresholds
2. Missing critical alerts due to channel failure
3. Monitoring blind spots

Runbook refs:
1. `AGENTS.md` (monitoring + emergency stop)
2. `docs/OPERATIONS.md`

### Skill: `backtest-analysis`
Purpose:
1. Run and evaluate backtesting/research workflows and identify gaps to live readiness.

Triggers:
1. "run backtest"
2. "evaluate strategy robustness"
3. "compare backtest vs live assumptions"

Inputs:
1. Backtesting configs/scripts
2. Historical data availability
3. Strategy parameter sets

Outputs:
1. Backtest results summary
2. Robustness notes and caveats
3. Follow-up tasks to close realism gaps

Dependencies:
1. `backtesting/*`
2. `scripts/run_backtest.py`

Failure modes:
1. Stubbed backtest modules used as if production-complete
2. Data quality/coverage issues
3. Unreproducible experiment settings

Runbook refs:
1. `README.md`
2. `AGENTS.md` (change control requirements)

### Skill: `release-change-control`
Purpose:
1. Enforce safe release practices for risk/execution/config changes.

Triggers:
1. "prepare release"
2. "deploy risk changes"
3. "switch to live"

Inputs:
1. Change diff
2. Test results
3. Environment target (`practice` or `live`)

Outputs:
1. Go/no-go checklist
2. Rollback plan
3. Approval record requirements

Dependencies:
1. Test suite status
2. Config and agent ownership

Failure modes:
1. Untested high-risk changes promoted
2. Missing rollback procedure
3. Live mode enabled without explicit approvals

Runbook refs:
1. `AGENTS.md` (safety policy + change control)
2. `docs/DEPLOYMENT.md`

## Skill Composition Rules
Recommended end-to-end chains:
1. Data issue chain:
   1. `market-data-ops` -> `strategy-validation` -> `risk-limit-audit`
2. Trade failure chain:
   1. `execution-reliability` -> `risk-limit-audit` -> `monitoring-incident-response`
3. Release chain:
   1. `backtest-analysis` -> `strategy-validation` -> `release-change-control`

Blocking dependencies:
1. Do not run `release-change-control` without current test results.
2. Do not approve `live` routing changes without `risk-limit-audit` and `execution-reliability`.

## Governance
When adding a skill:
1. Provide a unique `skill_id`.
2. Add at least three realistic trigger examples.
3. Define at least one concrete output artifact.
4. Link to runbook and source files.

When modifying a skill:
1. Update dependencies and failure modes first.
2. Validate that composition chains still make sense.

When deprecating a skill:
1. Mark it deprecated with replacement skill(s).
2. Remove it only after all references are migrated.

