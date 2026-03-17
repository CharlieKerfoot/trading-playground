"""Fill simulation from order book depth."""

from __future__ import annotations

from polymarket_playground.core.state import MarketState


class SlippageModel:
    """Slippage model that estimates fill price from book depth.

    Models:
    - Quadratic market impact (more realistic than linear)
    - Buy/sell asymmetry using bid/ask prices
    - Spread crossing cost (you pay the ask when buying, receive the bid when selling)
    - Partial spread capture for limit-order-like fills
    """

    def simulate_fill(
        self, direction: str, size: float, state: MarketState
    ) -> tuple[float, float]:
        """Simulate a fill and return (fill_price, spread_capture).

        Buying uses the ask side, so you pay more than mid.
        Market impact scales quadratically with size/depth.

        Directions:
            "yes"       — buying YES shares (pay ask)
            "no"        — buying NO shares (pay ask)
            "close_yes" — selling YES shares (receive bid)
            "close_no"  — selling NO shares (receive bid)

        Returns:
            (fill_price, spread_capture) tuple
        """
        is_buy = direction in ("yes", "no")

        if direction == "yes":
            base_price = state.yes_ask if state.yes_ask > 0 else state.yes_price
        elif direction == "no":
            base_price = state.no_ask if state.no_ask > 0 else state.no_price
        elif direction == "close_no":
            base_price = state.no_bid if state.no_bid > 0 else state.no_price
        else:
            # close_yes or legacy "close"
            base_price = state.yes_bid if state.yes_bid > 0 else state.yes_price

        # Quadratic market impact: slippage grows faster with larger orders
        book_depth = max(state.book_depth, 100.0)
        impact_ratio = size / book_depth
        slippage = base_price * impact_ratio * (1.0 + impact_ratio)

        if is_buy:
            fill_price = base_price + slippage
        else:
            fill_price = max(0.001, base_price - slippage)

        # Clamp to valid price range
        fill_price = max(0.001, min(0.999, fill_price))

        # Spread capture: small orders can capture a portion of the spread
        # Larger orders capture less due to market impact
        capture_ratio = max(0.0, 0.5 - impact_ratio)
        spread_capture = state.spread * size * capture_ratio

        return fill_price, spread_capture
