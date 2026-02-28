# Src Migration Status

## Goal
Move toward a professional `src/` package layout without breaking existing runtime commands and tests.

## Current Status
Completed:
1. Added canonical package path scaffold:
   1. `src/oanda_trading_system/`
   2. `src/oanda_trading_system/agents/`
   3. `src/oanda_trading_system/backtesting/`
   4. `src/oanda_trading_system/shared/`
2. Added compatibility alias modules that map `oanda_trading_system.*` imports to current legacy modules.
3. Added `src/oanda_trading_system/cli.py` mirror entrypoint.

Not yet migrated:
1. Physical move of `agents/`, `backtesting/`, `shared/` source files into `src/`.
2. Import rewrites across the codebase to use `oanda_trading_system.*` paths only.
3. Packaging metadata update to source-only distribution.

## Why This Stage Is Important
1. Zero runtime disruption while creating a stable target namespace.
2. Enables incremental module-by-module relocation.
3. Lets CI and tests validate each migration slice.

## Recommended Next Migration Steps
1. Move `shared/` first (smallest blast radius).
2. Update imports to `oanda_trading_system.shared.*`.
3. Add shim modules in legacy `shared/`.
4. Run full test suite.
5. Repeat for `backtesting/`, then `agents/`.

