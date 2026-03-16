"""Fill simulation from order book depth."""

from __future__ import annotations

from polymarket_playground.core.state import MarketState


class SlippageModel:
    """Simple slippage model that estimates fill price from book depth."""

    def simulate_fill(
        self, direction: str, size: float, state: MarketState
    ) -> tuple[float, float]:
        """Simulate a fill and return (fill_price, spread_capture).

        Args:
            direction: "yes" or "no"
            size: number of shares to fill
            state: current market snapshot

        Returns:
            (fill_price, spread_capture) tuple
        """
        if direction == "yes":
            mid = state.yes_price
        else:
            mid = state.no_price

        # Slippage scales with size relative to book depth
        book_depth = max(state.book_depth, 1e-6)  # avoid division by zero
        slippage = mid * (size / book_depth)
        fill_price = mid + slippage

        # Spread capture: limit-order-like fills capture half the spread
        spread_capture = state.spread * size * 0.5

        return fill_price, spread_capture
