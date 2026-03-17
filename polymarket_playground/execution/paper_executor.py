"""Paper (simulated) trade executor."""

from __future__ import annotations

from polymarket_playground.core.action import Action, DIRECTION_NAMES
from polymarket_playground.core.position import Position
from polymarket_playground.core.state import MarketState
from polymarket_playground.core.types import Fill
from polymarket_playground.execution.base_executor import BaseExecutor
from polymarket_playground.execution.slippage_model import SlippageModel


class PaperExecutor(BaseExecutor):
    """Simulates trade execution using a slippage model."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.fee_rate: float = config.get("fee_rate", 0.02)
        self.slippage_model = SlippageModel()

    def execute(
        self,
        action: Action,
        state: MarketState,
        position: Position | None = None,
    ) -> Fill | None:
        """Execute an action against current market state.

        Handles direction mapping:
        - hold (0): returns None
        - buy_yes (1): buy YES shares
        - buy_no (2): buy NO shares
        - close (3): close position at current prices
        """
        direction_name = DIRECTION_NAMES.get(action.direction, "hold")

        if direction_name == "hold":
            return None

        if direction_name == "close":
            return self._close(action, state, position)

        # buy_yes or buy_no
        token_direction = "yes" if direction_name == "buy_yes" else "no"
        return self._fill(token_direction, action.size, state)

    def _fill(self, direction: str, size: float, state: MarketState) -> Fill:
        """Simulate a fill for a given direction and size."""
        fill_price, spread_capture = self.slippage_model.simulate_fill(
            direction, size, state
        )
        fees = fill_price * size * self.fee_rate

        return Fill(
            direction=direction,
            size=size,
            price=fill_price,
            fees=fees,
            spread_capture=spread_capture,
            timestamp=state.timestamp,
        )

    def _close(
        self,
        action: Action,
        state: MarketState,
        position: Position | None = None,
    ) -> Fill:
        """Close position at current prices.

        Sells back the shares the agent holds. Uses the correct side
        (YES or NO) based on the actual position, or falls back to
        YES-side pricing when position info isn't available.
        """
        # Determine which side to sell based on the actual position
        sell_side = "close_yes"
        if position and position.no_shares > position.yes_shares:
            sell_side = "close_no"

        fill_price, spread_capture = self.slippage_model.simulate_fill(
            sell_side, action.size, state
        )
        fees = fill_price * action.size * self.fee_rate

        return Fill(
            direction="close",
            size=action.size,
            price=fill_price,
            fees=fees,
            spread_capture=spread_capture,
            timestamp=state.timestamp,
        )
