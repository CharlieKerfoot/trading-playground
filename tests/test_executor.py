"""Tests for PaperExecutor fill calculation."""

from datetime import datetime, timezone

from polymarket_playground.core.action import Action
from polymarket_playground.core.position import Position
from polymarket_playground.core.state import MarketState
from polymarket_playground.execution.paper_executor import PaperExecutor


def _make_state(**overrides) -> MarketState:
    defaults = dict(
        market_id="test",
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


def test_hold_returns_none():
    """Hold action produces no fill."""
    executor = PaperExecutor()
    state = _make_state()
    fill = executor.execute(Action(direction=0, size=0.0), state)
    assert fill is None


def test_buy_yes_fill():
    """Buy YES produces a fill with correct direction, size, and fees."""
    executor = PaperExecutor(config={"fee_rate": 0.02})
    state = _make_state()
    fill = executor.execute(Action(direction=1, size=5.0), state)

    assert fill is not None
    assert fill.direction == "yes"
    assert fill.size == 5.0
    # Fill price should be >= ask (slippage pushes it up)
    assert fill.price >= state.yes_ask
    # Fees should be price * size * rate
    expected_fees = fill.price * 5.0 * 0.02
    assert abs(fill.fees - expected_fees) < 1e-9


def test_buy_no_fill():
    """Buy NO produces a fill on the NO side."""
    executor = PaperExecutor()
    state = _make_state()
    fill = executor.execute(Action(direction=2, size=3.0), state)

    assert fill is not None
    assert fill.direction == "no"
    assert fill.size == 3.0
    assert fill.price >= state.no_ask


def test_close_fill():
    """Close action produces a fill with direction='close'."""
    executor = PaperExecutor()
    state = _make_state()
    position = Position()
    position.update(
        executor.execute(Action(direction=1, size=5.0), state)
    )

    close_fill = executor.execute(
        Action(direction=3, size=5.0), state, position
    )
    assert close_fill is not None
    assert close_fill.direction == "close"
    assert close_fill.size == 5.0
    # Close sells at bid, so price <= bid
    assert close_fill.price <= state.yes_bid + 0.01  # small slippage tolerance
