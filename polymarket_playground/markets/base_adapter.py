"""Abstract base class for market adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from polymarket_playground.core.state import MarketState
from polymarket_playground.core.types import Episode


class MarketAdapter(ABC):
    """Translates domain-specific data into the universal MarketState."""

    @abstractmethod
    def get_markets(self) -> list[str]:
        """Return the list of market IDs this adapter serves."""
        ...

    @abstractmethod
    def fetch_signals(self, market_id: str, timestamp: datetime) -> dict[str, float]:
        """Fetch external signals for *market_id* at *timestamp*."""
        ...

    @abstractmethod
    def build_agent_context(self, state: MarketState) -> str:
        """Build a human-readable context string from the current state."""
        ...

    @abstractmethod
    def select_episodes(self, filters: dict) -> list[Episode]:
        """Return episodes matching *filters*."""
        ...
