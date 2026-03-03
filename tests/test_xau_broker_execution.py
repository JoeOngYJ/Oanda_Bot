from __future__ import annotations

from scripts.xau_broker_execution import BrokerExecutionConfig, build_broker_setup_snapshot


def test_broker_setup_snapshot_basic_fields_and_cost_mapping():
    cfg = BrokerExecutionConfig(
        starting_equity=10_000.0,
        leverage=30.0,
        stop_out_margin_level_pct=40.0,
        max_margin_utilization_pct=0.85,
        spread_bps=1.0,
        slippage_bps=0.25,
        commission_bps_per_side=0.0,
    )
    snap = build_broker_setup_snapshot(cfg, ref_price=2000.0, stress_move_pct=0.01)
    assert snap["notional_cap"] == 255000.0
    assert abs(float(snap["roundtrip_spread_usd_per_oz"]) - 0.2) < 1e-12
    assert abs(float(snap["roundtrip_slippage_usd_per_oz"]) - 0.05) < 1e-12
    assert "warnings" in snap

