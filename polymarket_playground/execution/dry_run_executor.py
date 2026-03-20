"""Dry-run executor — logs what WOULD be sent without placing real orders."""

from __future__ import annotations

import logging

from polymarket_playground.core.action import Action, DIRECTION_NAMES
from polymarket_playground.core.position import Position
from polymarket_playground.core.state import MarketState
from polymarket_playground.core.types import Fill
from polymarket_playground.execution.base_executor import BaseExecutor
from polymarket_playground.execution.slippage_model import SlippageModel

logger = logging.getLogger(__name__)


class DryRunExecutor(BaseExecutor):
    """Executor that logs order details without sending them.

    Uses SlippageModel to simulate realistic fills so the rest of the
    system (positions, P&L tracking) works normally.
    """

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
        direction_name = DIRECTION_NAMES.get(action.direction, "hold")

        if direction_name == "hold":
            return None

        if direction_name == "close":
            return self._close(action, state, position)

        token_direction = "yes" if direction_name == "buy_yes" else "no"

        logger.info(
            "[DRY RUN] Would send order: direction=%s size=%.4f price=%.4f market_id=%s",
            token_direction,
            action.size,
            state.yes_ask if token_direction == "yes" else state.no_ask,
            state.market_id,
        )

        fill_price, spread_capture = self.slippage_model.simulate_fill(
            token_direction, action.size, state
        )
        fees = fill_price * action.size * self.fee_rate

        return Fill(
            direction=token_direction,
            size=action.size,
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
        sell_side = "close_yes"
        if position and position.no_shares > position.yes_shares:
            sell_side = "close_no"

        logger.info(
            "[DRY RUN] Would send close order: side=%s size=%.4f market_id=%s",
            sell_side,
            action.size,
            state.market_id,
        )

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
