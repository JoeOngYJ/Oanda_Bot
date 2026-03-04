#!/usr/bin/env python3
"""Tests for execution control features (kill-switch, shadow mode, idempotency)."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from oanda_bot.agents.execution.agent import ExecutionAgent
from oanda_bot.utils.config import Config
from oanda_bot.utils.models import ExecutionControlCommand, Instrument, RiskCheckResult, Side, TradeSignal


@pytest.fixture
def config(temp_config_dir):
    return Config.load(str(temp_config_dir))


@pytest.fixture
def sample_signal():
    return TradeSignal(
        signal_id="signal-exec-001",
        instrument=Instrument.EUR_USD,
        side=Side.BUY,
        quantity=1000,
        entry_price=Decimal("1.08500"),
        stop_loss=Decimal("1.08300"),
        take_profit=Decimal("1.08900"),
        confidence=0.8,
        rationale="Execution agent control test signal",
        strategy_name="TestStrategy",
        strategy_version="1.0.0",
        timestamp=datetime.utcnow(),
    )


@pytest.mark.asyncio
async def test_shadow_mode_executes_without_broker(config, sample_signal):
    agent = ExecutionAgent(config)
    agent.message_bus.publish = AsyncMock()
    agent.shadow_mode = True
    agent.kill_switch_active = False

    risk_check = RiskCheckResult(signal_id=sample_signal.signal_id, signal=sample_signal, approved=True)
    await agent._handle_approved_signal(risk_check)

    # Execution published and marked shadow.
    published_streams = [call.args[0] for call in agent.message_bus.publish.call_args_list]
    assert "stream:executions" in published_streams
    execution_payloads = [
        call.args[1]
        for call in agent.message_bus.publish.call_args_list
        if call.args[0] == "stream:executions"
    ]
    assert execution_payloads
    assert execution_payloads[-1]["execution_mode"] == "shadow"
    assert sample_signal.signal_id in agent._executed_signal_ids


@pytest.mark.asyncio
async def test_kill_switch_blocks_execution(config, sample_signal):
    agent = ExecutionAgent(config)
    agent.message_bus.publish = AsyncMock()
    agent.shadow_mode = True
    agent.kill_switch_active = True

    risk_check = RiskCheckResult(signal_id=sample_signal.signal_id, signal=sample_signal, approved=True)
    await agent._handle_approved_signal(risk_check)

    published_streams = [call.args[0] for call in agent.message_bus.publish.call_args_list]
    assert "stream:executions" not in published_streams
    assert sample_signal.signal_id not in agent._executed_signal_ids


@pytest.mark.asyncio
async def test_control_command_toggles_modes(config):
    agent = ExecutionAgent(config)
    agent.message_bus.publish = AsyncMock()

    await agent._apply_control_command(
        ExecutionControlCommand(action="kill_switch_on", reason="test", requested_by="pytest")
    )
    assert agent.kill_switch_active is True

    await agent._apply_control_command(
        ExecutionControlCommand(action="shadow_mode_on", reason="test", requested_by="pytest")
    )
    assert agent.shadow_mode is True

    await agent._apply_control_command(
        ExecutionControlCommand(action="kill_switch_off", reason="test", requested_by="pytest")
    )
    assert agent.kill_switch_active is False


@pytest.mark.asyncio
async def test_live_guardrail_blocks_without_explicit_enable(config, monkeypatch):
    monkeypatch.delenv("EXECUTION_LIVE_ENABLED", raising=False)
    monkeypatch.delenv("EXECUTION_SHADOW_MODE", raising=False)
    config.oanda.environment = "live"

    agent = ExecutionAgent(config)
    agent.message_bus.publish = AsyncMock()
    await agent._apply_startup_guardrails()

    assert agent.kill_switch_active is True
