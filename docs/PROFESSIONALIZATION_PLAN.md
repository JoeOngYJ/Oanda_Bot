# Professionalization Plan

This roadmap explains how to optimize the repository and make it production-grade.

## Objectives
1. Improve maintainability and developer speed.
2. Reduce runtime risk in risk/execution paths.
3. Enforce consistent data contracts and release discipline.
4. Improve observability and operational reliability.

## Phase 1 (Completed in this pass)
1. Documentation standardization:
   1. Root README rewritten.
   2. Feature catalog created.
   3. Repo structure and cleanup guide added.
   4. HTML executive overview added.
2. Backtesting execution cost realism:
   1. Spread-aware fills.
   2. Commission model variants.
   3. SL/TP lifecycle exits.
   4. Cost-focused test coverage.

## Phase 2 (High Priority)
1. Contract hardening:
   1. Align risk -> execution payloads with strict schema tests.
   2. Add contract tests for every Redis stream event type.
2. Config hardening:
   1. Split `practice` vs `live` profiles.
   2. Add required-field and safety-gate validation scripts.
3. CI baseline:
   1. Add GitHub Actions or equivalent for lint + tests.
   2. Block merges on failing risk/execution/backtesting tests.

## Phase 3 (Architecture and Packaging)
1. Migrate to clean `src/` package layout with backward compatibility shims.
2. Replace ad-hoc script execution with CLI commands (single entrypoint).
3. Introduce typed interfaces for strategy plugins and execution adapters.

## Phase 4 (Operational Excellence)
1. Add structured audit logs for all order lifecycle events.
2. Add SLOs/SLIs:
   1. Market data freshness
   2. Signal latency
   3. Risk decision latency
   4. Execution success/reject rates
3. Add incident templates and postmortem workflow.

## Phase 5 (Model and Strategy Governance)
1. Strategy versioning and artifact registry for backtests.
2. Reproducible experiment config snapshots.
3. Promotion gates from backtest -> paper -> live.

## Quality Gates to Enforce
1. No `live` config changes without dual approval.
2. Mandatory regression tests for risk and execution modifications.
3. Backtesting cost-model tests required for strategy-release PRs.
4. Docs update required for stream/model contract changes.

## Immediate Next 10 Tasks
1. Add stream contract integration tests for all event payloads.
2. Fix remaining risk/execution schema drift in live-agent pipeline.
3. Add typed DTO layer for stream payload compatibility checks.
4. Create `practice.yaml` and `live.yaml` config overlays.
5. Add pre-commit hooks (`ruff`, `black`, `mypy` where feasible).
6. Add CI pipeline with matrix test runs.
7. Add release checklist and changelog policy.
8. Add end-to-end replay backtests from stored historical data.
9. Improve backtesting metrics module beyond placeholder.
10. Add runtime dashboard panels tied to alert thresholds.

