"""Deterministic rule-based baseline agent."""

from __future__ import annotations

from polymarket_playground.core.action import Action
from polymarket_playground.core.state import MarketState

from .base_agent import BaseAgent


class RuleAgent(BaseAgent):
    """Simple threshold-based strategy for baseline comparison.

    Logic:
        - If ``yes_price < 0.3`` -> buy YES  (underpriced)
        - If ``yes_price > 0.7`` -> buy NO   (overpriced)
        - Otherwise              -> hold

    Size is always ``0.5``.
    """

    def act(self, state: MarketState) -> Action:
        if state.yes_price < 0.3:
            return Action(direction=1, size=0.5, reasoning="yes_price < 0.3 — buy YES")
        if state.yes_price > 0.7:
            return Action(direction=2, size=0.5, reasoning="yes_price > 0.7 — buy NO")
        return Action(direction=0, size=0.0, reasoning="price in neutral zone — hold")
