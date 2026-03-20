"""Tests for RiskGate safety limits."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import patch

from polymarket_playground.core.action import Action
from polymarket_playground.core.portfolio import Portfolio
from polymarket_playground.core.position import Position
from polymarket_playground.core.state import MarketState
from polymarket_playground.execution.risk_gate import RiskConfig, RiskGate


def _make_state(**overrides) -> MarketState:
    defaults = dict(
        market_id="test-market",
        yes_price=0.50,
        no_price=0.50,
        yes_bid=0.49,
        yes_ask=0.51,
        no_bid=0.49,
        no_ask=0.51,
        spread=0.02,
        book_depth=1000.0,
        timestamp=datetime.now(tz=timezone.utc),
    )
    defaults.update(overrides)
    return MarketState(**defaults)


def _make_portfolio(capital: float = 1000.0) -> Portfolio:
    return Portfolio(starting_capital=capital)


# -------------------------------------------------------------------
# Hold actions always pass through
# -------------------------------------------------------------------

def test_hold_always_allowed():
    gate = RiskGate()
    state = _make_state()
    portfolio = _make_portfolio()
    position = portfolio.get_position(state.market_id)
    action = Action(direction=0, size=0.0)
    result = gate.check(action, state, position, portfolio)
    assert result is action


# -------------------------------------------------------------------
# Kill switch
# -------------------------------------------------------------------

def test_kill_switch_blocks_all_actions():
    gate = RiskGate()
    gate._state.kill_switch = True
    state = _make_state()
    portfolio = _make_portfolio()
    position = portfolio.get_position(state.market_id)
    action = Action(direction=1, size=5.0)
    result = gate.check(action, state, position, portfolio)
    assert result is None
    assert len(gate.interventions) == 1
    assert "kill_switch" in gate.interventions[0]["reason"]


def test_kill_switch_triggered_by_session_loss():
    config = RiskConfig(max_loss_per_session_usd=10.0)
    gate = RiskGate(config)
    portfolio = _make_portfolio(capital=100.0)
    gate.init_session(portfolio)

    # Simulate a big loss by giving portfolio a losing position
    state = _make_state(yes_price=0.10)
    position = portfolio.get_position(state.market_id)
    position.yes_shares = 50.0
    position.cost_basis = 25.0  # bought at 0.50

    action = Action(direction=1, size=5.0)
    result = gate.check(action, state, position, portfolio)
    assert result is None
    assert gate.is_killed


# -------------------------------------------------------------------
# Per-market stop loss
# -------------------------------------------------------------------

def test_stop_loss_forces_close():
    config = RiskConfig(max_loss_per_market_usd=5.0)
    gate = RiskGate(config)
    portfolio = _make_portfolio(capital=100.0)
    gate.init_session(portfolio)

    state = _make_state(yes_price=0.10)
    position = portfolio.get_position(state.market_id)
    position.yes_shares = 20.0
    position.cost_basis = 10.0  # bought at 0.50, now worth 0.10 => loss = 2.0 - 10.0 = -8.0

    action = Action(direction=1, size=5.0)  # agent wants to buy more
    result = gate.check(action, state, position, portfolio)
    assert result is not None
    assert result.direction == 3  # forced close
    assert "stop loss" in result.reasoning


# -------------------------------------------------------------------
# Position size limit
# -------------------------------------------------------------------

def test_position_limit_blocks_buy():
    config = RiskConfig(max_position_usd=10.0)
    gate = RiskGate(config)
    portfolio = _make_portfolio(capital=100.0)
    gate.init_session(portfolio)

    state = _make_state()
    position = portfolio.get_position(state.market_id)
    position.cost_basis = 8.0  # already have $8 in position

    action = Action(direction=1, size=5.0)  # 5 * 0.51 = $2.55, total = $10.55 > $10
    result = gate.check(action, state, position, portfolio)
    assert result is None
    assert any("position_limit" in i["reason"] for i in gate.interventions)


# -------------------------------------------------------------------
# Total exposure limit
# -------------------------------------------------------------------

def test_exposure_limit_blocks_buy():
    config = RiskConfig(max_total_exposure_usd=15.0)
    gate = RiskGate(config)
    portfolio = _make_portfolio(capital=100.0)
    gate.init_session(portfolio)

    # Create existing exposure in another market
    other_pos = portfolio.get_position("other-market")
    other_pos.cost_basis = 12.0

    state = _make_state()
    position = portfolio.get_position(state.market_id)

    action = Action(direction=1, size=10.0)  # 10 * 0.51 = $5.10, total = $17.10 > $15
    result = gate.check(action, state, position, portfolio)
    assert result is None
    assert any("exposure_limit" in i["reason"] for i in gate.interventions)


# -------------------------------------------------------------------
# Capital check
# -------------------------------------------------------------------

def test_insufficient_capital_blocks_buy():
    config = RiskConfig()
    gate = RiskGate(config)
    portfolio = _make_portfolio(capital=1.0)  # only $1
    gate.init_session(portfolio)

    state = _make_state()
    position = portfolio.get_position(state.market_id)

    action = Action(direction=1, size=10.0)  # 10 * 0.51 = $5.10 > $1
    result = gate.check(action, state, position, portfolio)
    assert result is None
    assert any("insufficient_capital" in i["reason"] for i in gate.interventions)


# -------------------------------------------------------------------
# Rate limiting
# -------------------------------------------------------------------

def test_rate_limit_blocks_excess_orders():
    config = RiskConfig(max_orders_per_minute=3)
    gate = RiskGate(config)
    portfolio = _make_portfolio(capital=1000.0)
    gate.init_session(portfolio)

    state = _make_state()
    position = portfolio.get_position(state.market_id)

    # First 3 orders should pass
    for _ in range(3):
        action = Action(direction=1, size=1.0)
        result = gate.check(action, state, position, portfolio)
        assert result is not None

    # 4th should be blocked
    action = Action(direction=1, size=1.0)
    result = gate.check(action, state, position, portfolio)
    assert result is None
    assert any("rate_limit" in i["reason"] for i in gate.interventions)


# -------------------------------------------------------------------
# Cooldown after close
# -------------------------------------------------------------------

def test_cooldown_after_close():
    config = RiskConfig(cooldown_after_close_seconds=60.0)
    gate = RiskGate(config)
    portfolio = _make_portfolio(capital=1000.0)
    gate.init_session(portfolio)

    state = _make_state()
    position = portfolio.get_position(state.market_id)
    position.yes_shares = 5.0
    position.cost_basis = 2.5

    # Close action records timestamp
    close_action = Action(direction=3, size=5.0)
    result = gate.check(close_action, state, position, portfolio)
    assert result is not None

    # Immediate buy should be blocked by cooldown
    buy_action = Action(direction=1, size=1.0)
    position_after_close = portfolio.get_position(state.market_id)
    result = gate.check(buy_action, state, position_after_close, portfolio)
    assert result is None
    assert any("cooldown_after_close" in i["reason"] for i in gate.interventions)


# -------------------------------------------------------------------
# Close action allowed when position is losing
# -------------------------------------------------------------------

def test_close_always_allowed_regardless_of_capital():
    config = RiskConfig()
    gate = RiskGate(config)
    portfolio = _make_portfolio(capital=0.0)  # no capital
    gate.init_session(portfolio)

    state = _make_state()
    position = portfolio.get_position(state.market_id)
    position.yes_shares = 5.0
    position.cost_basis = 2.5

    close_action = Action(direction=3, size=5.0)
    result = gate.check(close_action, state, position, portfolio)
    assert result is not None
    assert result.direction == 3


# -------------------------------------------------------------------
# Interventions tracking
# -------------------------------------------------------------------

def test_interventions_accumulated():
    gate = RiskGate()
    gate._state.kill_switch = True
    state = _make_state()
    portfolio = _make_portfolio()
    position = portfolio.get_position(state.market_id)

    for _ in range(3):
        gate.check(Action(direction=1, size=1.0), state, position, portfolio)

    assert len(gate.interventions) == 3
    for i in gate.interventions:
        assert "market_id" in i
        assert "reason" in i
        assert "timestamp" in i


# -------------------------------------------------------------------
# Buy NO also checked
# -------------------------------------------------------------------

def test_buy_no_position_limit():
    config = RiskConfig(max_position_usd=5.0)
    gate = RiskGate(config)
    portfolio = _make_portfolio(capital=100.0)
    gate.init_session(portfolio)

    state = _make_state()
    position = portfolio.get_position(state.market_id)
    position.cost_basis = 4.0

    action = Action(direction=2, size=5.0)  # buy NO: 5 * 0.51 = $2.55, total $6.55 > $5
    result = gate.check(action, state, position, portfolio)
    assert result is None
