"""Abstract base agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from polymarket_playground.core.action import Action
from polymarket_playground.core.state import MarketState
from polymarket_playground.core.types import EpisodeResult


class BaseAgent(ABC):
    """Base class for all trading agents.

    Every agent — RL policies, Claude, rule-based, human — implements this
    interface.  Agents receive the full ``MarketState`` and return an ``Action``.
    """

    @abstractmethod
    def act(self, state: MarketState) -> Action:
        """Choose an action given the current market state."""
        ...

    def on_reset(self, state: MarketState) -> None:
        """Called at the start of each episode.  Override for stateful agents."""
        pass

    def on_episode_end(self, result: EpisodeResult) -> None:
        """Called at resolution.  Override for learning agents."""
        pass
