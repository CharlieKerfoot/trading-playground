"""Deterministic rule-based baseline agent."""

from __future__ import annotations

from polymarket_playground.core.action import Action
from polymarket_playground.core.state import MarketState

from .base_agent import BaseAgent


class RuleAgent(BaseAgent):
    """Threshold-based strategy with position management.

    Logic:
        - Buy YES when price < 0.35 (likely mispriced, edge on YES)
        - Buy NO when price > 0.65 (likely mispriced, edge on NO)
        - Scale into positions gradually (don't go all-in immediately)
        - Close position near resolution or when profitable
        - Avoid overtrading in the neutral zone
    """

    def __init__(
        self,
        buy_threshold: float = 0.35,
        sell_threshold: float = 0.65,
        close_time: float = 0.85,
    ) -> None:
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.close_time = close_time

    def act(self, state: MarketState) -> Action:
        yes = state.yes_price
        has_yes = state.agent_yes_shares > 0.01
        has_no = state.agent_no_shares > 0.01
        has_position = has_yes or has_no

        # Close position near end of episode to lock in profits
        if has_position and state.time_elapsed > self.close_time:
            return Action(direction=3, size=1.0, reasoning="closing near resolution")

        # Close if position is profitable and price moved to neutral zone
        if has_position and 0.4 < yes < 0.6:
            unrealized = (
                state.agent_yes_shares * yes
                + state.agent_no_shares * (1.0 - yes)
                - state.agent_cost_basis
            )
            if unrealized > 0.02:
                return Action(direction=3, size=1.0, reasoning="taking profit in neutral zone")

        # Don't buy more if already holding the opposite side
        if yes < self.buy_threshold and not has_no:
            edge = 0.5 - yes
            size = min(1.0, edge * 3.0)
            return Action(direction=1, size=size, reasoning=f"yes={yes:.2f} buy YES (edge={edge:.2f})")

        if yes > self.sell_threshold and not has_yes:
            edge = yes - 0.5
            size = min(1.0, edge * 3.0)
            return Action(direction=2, size=size, reasoning=f"yes={yes:.2f} buy NO (edge={edge:.2f})")

        return Action(direction=0, size=0.0, reasoning="neutral zone — hold")
