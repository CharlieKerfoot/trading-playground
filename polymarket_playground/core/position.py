"""Position tracking, P&L, and settlement."""

from __future__ import annotations

from polymarket_playground.core.types import Fill


class Position:
    """Tracks a trading position in a binary market."""

    def __init__(self):
        self.yes_shares: float = 0.0
        self.no_shares: float = 0.0
        self.cost_basis: float = 0.0
        self.realized_pnl: float = 0.0
        self._prev_pnl: float = 0.0

    @property
    def net_exposure(self) -> float:
        """Positive = long YES, negative = long NO."""
        return self.yes_shares - self.no_shares

    @property
    def is_flat(self) -> bool:
        return self.yes_shares == 0.0 and self.no_shares == 0.0

    def update(self, fill: Fill | None):
        """Update position from a fill."""
        if fill is None:
            return
        if fill.direction == "yes":
            self.yes_shares += fill.size
            self.cost_basis += fill.size * fill.price
        elif fill.direction == "no":
            self.no_shares += fill.size
            self.cost_basis += fill.size * fill.price
        elif fill.direction == "close":
            # Realize P&L and flatten position
            close_value = (
                self.yes_shares * fill.price
                + self.no_shares * (1.0 - fill.price)
            )
            self.realized_pnl += close_value - self.cost_basis
            self.yes_shares = 0.0
            self.no_shares = 0.0
            self.cost_basis = 0.0

    def mtm_delta(self, current_yes_price: float) -> float:
        """Mark-to-market P&L change since last call."""
        current_mtm = (
            self.yes_shares * current_yes_price
            + self.no_shares * (1.0 - current_yes_price)
        )
        current_pnl = current_mtm - self.cost_basis + self.realized_pnl
        delta = current_pnl - self._prev_pnl
        self._prev_pnl = current_pnl
        return delta

    def settlement_pnl(self, resolution: float | None) -> float:
        """P&L at market resolution. resolution=1.0 means YES wins."""
        if resolution is None:
            return 0.0
        yes_value = self.yes_shares * resolution
        no_value = self.no_shares * (1.0 - resolution)
        return yes_value + no_value - self.cost_basis

    def close_value(self, current_yes_price: float) -> float:
        """Value if position were closed at current prices."""
        return (
            self.yes_shares * current_yes_price
            + self.no_shares * (1.0 - current_yes_price)
        )
