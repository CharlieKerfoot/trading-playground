"""Abstract base class for trade execution."""

from __future__ import annotations

from abc import ABC, abstractmethod

from polymarket_playground.core.action import Action
from polymarket_playground.core.state import MarketState
from polymarket_playground.core.types import Fill


class BaseExecutor(ABC):
    @abstractmethod
    def execute(self, action: Action, state: MarketState) -> Fill | None:
        """Execute an action against current market state. Returns Fill or None for hold."""
        ...
