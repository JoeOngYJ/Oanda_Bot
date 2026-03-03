from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class BrokerExecutionConfig:
    starting_equity: float = 10_000.0
    leverage: float = 30.0
    stop_out_margin_level_pct: float = 40.0
    max_margin_utilization_pct: float = 0.95
    spread_bps: float = 2.0
    slippage_bps: float = 0.5
    commission_bps_per_side: float = 0.0


def build_broker_setup_snapshot(
    cfg: BrokerExecutionConfig,
    ref_price: float = 2000.0,
    stress_move_pct: float = 0.01,
) -> Dict[str, float | bool | str]:
    """Deterministic sanity snapshot for leverage/margin/cost realism checks."""
    eq = float(cfg.starting_equity)
    lev = max(1e-9, float(cfg.leverage))
    mmu = float(np.clip(cfg.max_margin_utilization_pct, 0.0, 1.0))
    notional_cap = max(0.0, eq * lev * mmu)
    used_margin = notional_cap / lev
    margin_level_at_entry_pct = float("inf") if used_margin <= 0 else (eq / used_margin) * 100.0

    # Approximate one-bar stress loss at full utilization (before stop-out liquidation effects).
    stress_notional_loss = notional_cap * float(max(0.0, stress_move_pct))
    stress_equity = eq - stress_notional_loss
    stress_margin_level_pct = float("inf") if used_margin <= 0 else (stress_equity / used_margin) * 100.0
    stopout_breach_under_stress = bool(stress_margin_level_pct < float(cfg.stop_out_margin_level_pct))

    px = float(max(1e-9, ref_price))
    roundtrip_spread_usd = px * (float(cfg.spread_bps) * 1e-4)
    roundtrip_slippage_usd = px * (float(cfg.slippage_bps) * 1e-4)
    roundtrip_commission_usd_per_1x_notional = 2.0 * float(cfg.commission_bps_per_side) * 1e-4

    warnings: List[str] = []
    if mmu > 0.90:
        warnings.append("max_margin_utilization_pct > 0.90 can force stop-outs in routine volatility.")
    if float(cfg.stop_out_margin_level_pct) < 40.0:
        warnings.append("stop_out_margin_level_pct below 40 may understate broker liquidation risk.")
    if roundtrip_spread_usd > 0.50:
        warnings.append("spread_bps implies >$0.50/oz round-trip spread at reference price.")

    return {
        "starting_equity": eq,
        "leverage": lev,
        "max_margin_utilization_pct": mmu,
        "stop_out_margin_level_pct": float(cfg.stop_out_margin_level_pct),
        "notional_cap": float(notional_cap),
        "used_margin_at_cap": float(used_margin),
        "margin_level_at_entry_pct": float(margin_level_at_entry_pct),
        "stress_move_pct": float(stress_move_pct),
        "equity_after_stress": float(stress_equity),
        "margin_level_after_stress_pct": float(stress_margin_level_pct),
        "stopout_breach_under_stress": stopout_breach_under_stress,
        "reference_price": px,
        "roundtrip_spread_usd_per_oz": float(roundtrip_spread_usd),
        "roundtrip_slippage_usd_per_oz": float(roundtrip_slippage_usd),
        "roundtrip_commission_per_notional_frac": float(roundtrip_commission_usd_per_1x_notional),
        "warnings": warnings,
    }


def _price_with_entry_cost(price: float, side: int, spread_bps: float, slippage_bps: float) -> float:
    half = 0.5 * (spread_bps + slippage_bps) * 1e-4
    if side > 0:
        return price * (1.0 + half)
    return price * (1.0 - half)


def _price_with_exit_cost(price: float, side: int, spread_bps: float, slippage_bps: float) -> float:
    half = 0.5 * (spread_bps + slippage_bps) * 1e-4
    if side > 0:
        return price * (1.0 - half)
    return price * (1.0 + half)


def _commission_cash(notional: float, commission_bps_per_side: float) -> float:
    return abs(notional) * (commission_bps_per_side * 1e-4) * 2.0


def simulate_one_bar_portfolio_step(
    equity_before: float,
    close_t: float,
    high_t: float,
    low_t: float,
    close_next: float,
    sleeve_positions: List[Dict[str, float]],
    cfg: BrokerExecutionConfig,
) -> Dict[str, object]:
    """Simulate one-bar broker-style leveraged PnL with stop-out checks.

    sleeve_positions items:
      {"sleeve": str, "side": {-1,+1}, "allocation": float}
    """
    eq0 = float(equity_before)
    if eq0 <= 0.0 or not sleeve_positions:
        return {
            "equity_before": eq0,
            "equity_after": eq0,
            "pnl_cash": 0.0,
            "used_margin": 0.0,
            "margin_level_pct": float("inf"),
            "stopout_triggered": False,
            "per_sleeve_pnl_cash": {},
        }

    alloc = np.array([max(0.0, float(p.get("allocation", 0.0))) for p in sleeve_positions], dtype=float)
    sides = np.array([int(np.sign(float(p.get("side", 0.0)))) for p in sleeve_positions], dtype=int)
    names = [str(p.get("sleeve", f"sleeve_{i}")) for i, p in enumerate(sleeve_positions)]

    valid = (alloc > 0.0) & (sides != 0)
    if not np.any(valid):
        return {
            "equity_before": eq0,
            "equity_after": eq0,
            "pnl_cash": 0.0,
            "used_margin": 0.0,
            "margin_level_pct": float("inf"),
            "stopout_triggered": False,
            "per_sleeve_pnl_cash": {},
        }
    alloc = alloc[valid]
    sides = sides[valid]
    names = [n for i, n in enumerate(names) if valid[i]]

    # Convert allocation shares into notional exposure under leverage and margin limits.
    total_alloc = float(np.sum(alloc))
    notional_cap = max(0.0, eq0 * float(cfg.leverage) * float(cfg.max_margin_utilization_pct))
    notionals = notional_cap * (alloc / max(1e-12, total_alloc))
    used_margin = float(np.sum(notionals / max(1e-12, float(cfg.leverage))))

    # Entry / exit prices include spread+slippage adverse adjustments.
    entry = np.array([_price_with_entry_cost(float(close_t), int(s), cfg.spread_bps, cfg.slippage_bps) for s in sides], dtype=float)
    exit_ = np.array([_price_with_exit_cost(float(close_next), int(s), cfg.spread_bps, cfg.slippage_bps) for s in sides], dtype=float)
    units = notionals / np.maximum(np.abs(entry), 1e-12)
    gross = np.where(sides > 0, units * (exit_ - entry), units * (entry - exit_))
    comm = np.array([_commission_cash(float(n), cfg.commission_bps_per_side) for n in notionals], dtype=float)
    net = gross - comm

    # Worst-case intrabar move for stop-out check.
    worst_exit_px = np.array(
        [
            _price_with_exit_cost(float(low_t if s > 0 else high_t), int(s), cfg.spread_bps, cfg.slippage_bps)
            for s in sides
        ],
        dtype=float,
    )
    worst_gross = np.where(sides > 0, units * (worst_exit_px - entry), units * (entry - worst_exit_px))
    worst_net = worst_gross - comm
    equity_worst = eq0 + float(np.sum(worst_net))
    margin_level = float("inf") if used_margin <= 0 else (equity_worst / used_margin) * 100.0

    stopout = bool(margin_level < float(cfg.stop_out_margin_level_pct))
    pnl_vec = worst_net if stopout else net
    pnl_cash = float(np.sum(pnl_vec))
    eq1 = eq0 + pnl_cash

    per = {n: float(v) for n, v in zip(names, pnl_vec)}
    return {
        "equity_before": eq0,
        "equity_after": float(eq1),
        "pnl_cash": pnl_cash,
        "used_margin": used_margin,
        "margin_level_pct": margin_level,
        "stopout_triggered": stopout,
        "per_sleeve_pnl_cash": per,
    }
