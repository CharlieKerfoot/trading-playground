"""Tests for Position tracking, P&L, and settlement."""

from polymarket_playground.core.position import Position
from polymarket_playground.core.types import Fill


def test_buy_yes_updates_position():
    """Buying YES shares increases yes_shares and cost_basis."""
    pos = Position()
    fill = Fill(direction="yes", size=5.0, price=0.60)
    pos.update(fill)

    assert pos.yes_shares == 5.0
    assert pos.no_shares == 0.0
    assert pos.cost_basis == 3.0  # 5 * 0.60
    assert pos.is_flat is False


def test_buy_no_updates_position():
    """Buying NO shares increases no_shares and cost_basis."""
    pos = Position()
    fill = Fill(direction="no", size=3.0, price=0.40)
    pos.update(fill)

    assert pos.no_shares == 3.0
    assert pos.yes_shares == 0.0
    assert abs(pos.cost_basis - 1.2) < 1e-9  # 3 * 0.40
    assert pos.net_exposure == -3.0  # long NO


def test_close_realizes_pnl():
    """Closing a position moves value to realized_pnl and flattens."""
    pos = Position()
    # Buy 5 YES at 0.40 (cost = 2.0)
    pos.update(Fill(direction="yes", size=5.0, price=0.40))

    # Close at 0.70 (close_value = 5 * 0.70 = 3.5, pnl = 3.5 - 2.0 = 1.5)
    pos.update(Fill(direction="close", size=5.0, price=0.70))

    assert pos.is_flat
    assert pos.yes_shares == 0.0
    assert pos.cost_basis == 0.0
    assert abs(pos.realized_pnl - 1.5) < 1e-9


def test_settlement_pnl():
    """Settlement P&L correctly values shares at resolution price."""
    pos = Position()
    # Buy 10 YES at 0.30 (cost = 3.0)
    pos.update(Fill(direction="yes", size=10.0, price=0.30))

    # YES wins: each share worth 1.0, total = 10.0, pnl = 10.0 - 3.0 = 7.0
    assert abs(pos.settlement_pnl(1.0) - 7.0) < 1e-9

    # NO wins: each YES share worth 0.0, pnl = 0.0 - 3.0 = -3.0
    assert abs(pos.settlement_pnl(0.0) - (-3.0)) < 1e-9

    # None resolution: no settlement
    assert pos.settlement_pnl(None) == 0.0


def test_mtm_delta():
    """Mark-to-market delta tracks P&L changes between calls."""
    pos = Position()
    pos.update(Fill(direction="yes", size=10.0, price=0.50))

    # First call at 0.50: mtm = 10*0.50 = 5.0, pnl = 5.0 - 5.0 = 0.0
    delta1 = pos.mtm_delta(0.50)
    assert abs(delta1 - 0.0) < 1e-9

    # Price rises to 0.60: mtm = 10*0.60 = 6.0, pnl = 6.0 - 5.0 = 1.0
    # delta = 1.0 - 0.0 = 1.0
    delta2 = pos.mtm_delta(0.60)
    assert abs(delta2 - 1.0) < 1e-9

    # Price drops to 0.55: mtm = 10*0.55 = 5.5, pnl = 5.5 - 5.0 = 0.5
    # delta = 0.5 - 1.0 = -0.5
    delta3 = pos.mtm_delta(0.55)
    assert abs(delta3 - (-0.5)) < 1e-9
