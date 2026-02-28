# Repository Structure and Cleanup

## Current Practical Layout
This structure is kept non-breaking so imports and scripts continue to work.

1. `agents/`
   1. Runtime agents: market data, strategy, risk, execution, monitoring.
2. `shared/`
   1. Common contracts and infrastructure: models, message bus, config, logging.
3. `backtesting/`
   1. Research/backtest framework and execution simulation.
4. `config/`
   1. Runtime and backtest configuration.
5. `scripts/`
   1. Operational helpers and runners.
6. `tests/`
   1. Unit, integration, backtesting, performance, stress, failover suites.
7. `docs/`
   1. Architecture, operations, deployment, feature catalog, roadmap.
8. `data/`
   1. Historical/backtest data artifacts.
9. `src/oanda_trading_system/`
   1. Canonical target namespace for phased migration.

## What Was Restructured
1. Documentation hub standardized and expanded.
2. Root README rewritten to reflect actual current capabilities.
3. Feature inventory formalized.
4. Professionalization plan documented.
5. HTML project summary added for quick non-technical review.
6. Added `src/` migration scaffold with compatibility alias modules.

## Why We Did Not Move Core Code Directories Yet
1. Existing imports and scripts rely on current paths.
2. Large directory moves would require broad regression testing and migration patches.
3. Current improvement focus is non-breaking professionalism first, then controlled codebase migration.

## Recommended Phase-2 Physical Refactor (Planned)
Target high-level layout:
1. `src/oanda_trading_system/`
   1. `agents/`
   2. `shared/`
   3. `backtesting/`
2. `configs/` for environment-specific profiles.
3. `tools/` for operational scripts and local utilities.
4. `docs/` for architecture, ops, ADRs, and runbooks.
5. `tests/` mirrored with `src` package boundaries.

## Migration Rules for Future Structural Moves
1. Move one subsystem at a time.
2. Add import shims and deprecation window.
3. Update scripts and tests in same change set.
4. Run full test suite before merge.
5. Update all docs and runbooks per migration.
