"""Portfolio — tracks capital and positions across multiple markets."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from polymarket_playground.core.position import Position
from polymarket_playground.core.types import Fill

logger = logging.getLogger(__name__)


@dataclass
class Portfolio:
  """Multi-market portfolio with capital tracking."""

  starting_capital: float
  _capital: float = field(init=False)
  _positions: dict[str, Position] = field(default_factory=dict, init=False)
  _realized_pnl: float = field(default=0.0, init=False)

  def __post_init__(self):
    self._capital = self.starting_capital

  @property
  def available_capital(self) -> float:
    return self._capital

  @property
  def active_markets(self) -> list[str]:
    return [mid for mid, pos in self._positions.items() if not pos.is_flat]

  def get_position(self, market_id: str) -> Position:
    if market_id not in self._positions:
      self._positions[market_id] = Position()
    return self._positions[market_id]

  def total_exposure(self) -> float:
    """Sum of cost basis across all open positions."""
    return sum(pos.cost_basis for pos in self._positions.values())

  def total_unrealized_pnl(self, prices: dict[str, float]) -> float:
    """Unrealized P&L given current yes prices per market."""
    pnl = 0.0
    for mid, pos in self._positions.items():
      if pos.is_flat:
        continue
      price = prices.get(mid, 0.5)
      pnl += pos.close_value(price) - pos.cost_basis
    return pnl

  def total_pnl(self, prices: dict[str, float]) -> float:
    """Total P&L (realized + unrealized)."""
    return self._realized_pnl + self.total_unrealized_pnl(prices)

  def update(self, market_id: str, fill: Fill | None) -> None:
    """Update portfolio from a fill."""
    if fill is None:
      return
    pos = self.get_position(market_id)
    old_realized = pos.realized_pnl
    old_cost_basis = pos.cost_basis

    if fill.direction in ("yes", "no"):
      # Deduct cost of opening a position
      cost = fill.size * fill.price + fill.fees
      self._capital -= cost
    elif fill.direction == "close":
      # On close, position returns close_value to capital
      close_value = (
        pos.yes_shares * fill.price
        + pos.no_shares * (1.0 - fill.price)
      )
      self._capital += close_value - fill.fees

    pos.update(fill)

    # Track realized P&L from closes
    realized_delta = pos.realized_pnl - old_realized
    if realized_delta != 0:
      self._realized_pnl += realized_delta

  def settle(self, market_id: str, resolution: float) -> float:
    """Settle a resolved market. Returns settlement P&L for that market."""
    pos = self.get_position(market_id)
    if pos.is_flat:
      return 0.0
    pnl = pos.settlement_pnl(resolution)
    # Return capital: settlement value
    settlement_value = (
      pos.yes_shares * resolution + pos.no_shares * (1.0 - resolution)
    )
    self._capital += settlement_value
    self._realized_pnl += pnl
    # Reset position
    self._positions[market_id] = Position()
    logger.info(
      "Settled market %s: resolution=%.1f pnl=%+.4f",
      market_id, resolution, pnl,
    )
    return pnl

  def suggest_size(self, confidence: float, max_fraction: float = 0.1) -> float:
    """Suggest position size as dollar amount.

    Uses a simplified Kelly-inspired approach: bet a fraction of available
    capital proportional to edge, capped at max_fraction.
    """
    if confidence <= 0.5:
      return 0.0
    edge = confidence - 0.5  # 0 to 0.5
    kelly_fraction = min(edge * 2, max_fraction)  # scale edge, cap at max
    return self._capital * kelly_fraction

  def snapshot(self) -> dict:
    """Return a serializable snapshot of the portfolio state."""
    positions = {}
    for mid, pos in self._positions.items():
      if not pos.is_flat:
        positions[mid] = {
          "yes_shares": pos.yes_shares,
          "no_shares": pos.no_shares,
          "cost_basis": pos.cost_basis,
          "realized_pnl": pos.realized_pnl,
        }
    return {
      "starting_capital": self.starting_capital,
      "available_capital": self._capital,
      "total_exposure": self.total_exposure(),
      "realized_pnl": self._realized_pnl,
      "active_positions": positions,
    }
