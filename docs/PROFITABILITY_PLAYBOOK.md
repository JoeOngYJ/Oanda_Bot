# Profitability Playbook

This playbook defines the operating standard to move from “feature complete” to “consistently profitable”.

## 1. Goal
Build a repeatable process that produces:
1. Positive expectancy after all costs.
2. Controlled drawdowns.
3. Reliable live operations.

## 2. Non-Negotiable Principles
1. No strategy promotion based on in-sample performance only.
2. All performance claims must be net of spread, slippage, and fees.
3. Risk of ruin control has priority over return maximization.
4. If live deviates materially from paper, reduce size or halt.

## 3. Metrics That Matter
Track these per strategy and portfolio:
1. `expectancy_per_trade` (net)
2. `profit_factor`
3. `win_rate`
4. `max_drawdown_pct`
5. `sharpe_like`
6. `avg_slippage_pips`
7. `order_rejection_rate`
8. `latency_signal_to_execution_ms`

## 4. Promotion Gates

### Gate A: Backtest Qualification
Minimum requirements:
1. Out-of-sample net expectancy > 0.
2. Profit factor >= 1.20.
3. Max drawdown <= 15%.
4. Parameter sensitivity acceptable:
   1. Small parameter changes do not collapse performance.
5. Regime robustness:
   1. Profitable or near-breakeven in at least 2 major market regimes.

Fail conditions:
1. Performance depends on a narrow parameter window.
2. Returns disappear after realistic transaction costs.
3. Single-period overfit dominates total result.

### Gate B: Paper Trading Qualification (4-8 Weeks)
Minimum requirements:
1. Net expectancy remains positive.
2. Paper slippage within 25% of modeled assumptions.
3. Rejection rate <= 1%.
4. No unresolved critical incidents.
5. Strategy behavior matches design assumptions.

Fail conditions:
1. Frequent contract/runtime errors.
2. Slippage materially worse than backtest assumptions.
3. Persistent strategy drift versus expected behavior.

### Gate C: Live Qualification (Micro Size)
Minimum requirements:
1. Start at smallest practical size.
2. Daily max loss and weekly max loss hard limits active.
3. Circuit breaker confirmed operational.
4. Live metrics remain within tolerance band versus paper:
   1. Expectancy delta <= 30%
   2. Slippage delta <= 30%
5. At least 100-200 live trades before size increase decisions.

Fail conditions:
1. Any risk control bypass.
2. Live expectancy turns significantly negative.
3. Operational instability (alerts, disconnects, missed fills).

## 5. Sizing and Risk Framework
1. Per-trade risk target:
   1. 0.25% to 0.75% of equity while maturing.
2. Portfolio heat limit:
   1. Max simultaneous aggregate risk <= 3%.
3. Correlation control:
   1. Cap correlated pair exposure.
4. Loss limits:
   1. Daily stop: 2% equity.
   2. Weekly stop: 4% equity.
5. De-risk rules:
   1. If rolling 20-trade expectancy < 0, cut size 50%.
   2. If rolling 50-trade expectancy < 0, halt strategy.

## 6. Execution Quality Standards
1. Every order path must be observable:
   1. signal time
   2. risk decision time
   3. submit time
   4. fill time
2. Track and alert:
   1. latency spikes
   2. rejection spikes
   3. spread spikes
3. Model and update real slippage weekly from logs.

## 7. Weekly Operating Cadence
1. Monday:
   1. Review prior week KPIs by strategy.
   2. Recompute rolling expectancy/drawdown windows.
2. Mid-week:
   1. Review incident logs and execution anomalies.
3. Friday:
   1. Strategy scorecard and promotion/demotion decisions.
   2. Update assumptions for spread/slippage/fees.
4. Use templates:
   1. `docs/STRATEGY_SCORECARD_TEMPLATE.md`
   2. `data/templates/strategy_scorecard_template.csv`

## 8. Strategy Lifecycle Rules
1. New strategy enters as `research`.
2. Must pass Gate A to become `paper`.
3. Must pass Gate B to become `live_micro`.
4. Must pass Gate C to become `live_scaled`.
5. Any strategy can be demoted on risk or performance breach.

## 9. Immediate Focus (Next 30 Days)
1. Harden strategy->risk->execution payload contracts.
2. Add live-paper-backtest KPI dashboard parity.
3. Run one strategy end-to-end through Gate B.
4. Establish weekly governance ritual and decision log.
5. No live scaling until all gates are met.

## 10. Decision Checklist (Before Any Live Size Increase)
1. Are rolling expectancy and drawdown within targets?
2. Is execution quality stable (slippage/rejections/latency)?
3. Did any risk controls misfire or get bypassed?
4. Are recent results consistent with paper expectation bands?
5. Is there a rollback plan if next week underperforms?

## 11. Scorecard Storage Convention
1. Keep completed weekly scorecards in:
   1. `data/scorecards/YYYY/`
2. Filename format:
   1. `YYYY-Www_<strategy_name>_<env>.csv`
3. Keep an append-only decision log for auditability.

## 12. Scorecard Automation Script
Create a new weekly scorecard file:
```bash
python scripts/scorecard_new_week.py \
  --strategy MA_Crossover_EURUSD \
  --environment paper \
  --reviewer ops_lead \
  --version 1.0.0
```

This writes:
1. `data/scorecards/YYYY/YYYY-Www_<strategy_name>_<env>.csv`

Aggregate scorecards into a portfolio KPI report:
```bash
python scripts/scorecard_report.py
```

This writes:
1. `data/reports/portfolio_kpi_report_<date-or-week>.md`
2. `data/reports/portfolio_kpi_report_<date-or-week>.csv`
