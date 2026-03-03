# GPU Runtime Refactor TODO

## Current state
- Runtime backtest now supports `--gpu {auto,on,off}`.
- Feature engineering and regime distance math can run on CuPy when available.
- Execution simulator, order fills, risk limits, and financing remain CPU (intentional for correctness).

## Priority next steps
1. Add batched feature cache (`M15/H1/H4/D1`) so we stop rebuilding arrays from deques each bar.
2. Add optional batched regime inference (`N x K` distance) per chunk to reduce Python loop overhead.
3. Profile hot paths (`Backtester._step`, strategy `on_bar`, simulator fills) with 2y M15 XAU runs.
4. Move pure math in risk sizing/guardrail checks to vectorized NumPy where possible.
5. Add benchmark script comparing `cpu` vs `gpu` wall-clock on identical windows.

## Deployment guardrails
- Keep PnL, fees, financing, and drawdown identical between `--gpu off` and `--gpu on` within floating tolerance.
- Fail fast on `--gpu on` when CUDA/CuPy is missing.
- Keep `--gpu auto` default for safe fallback.
