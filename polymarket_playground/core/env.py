"""TradingEnv — market-agnostic, agent-agnostic Gymnasium environment."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from polymarket_playground.core.action import Action
from polymarket_playground.core.position import Position
from polymarket_playground.core.state import MarketState, ObservationBuilder
from polymarket_playground.core.types import Fill
from polymarket_playground.data.historical_loader import HistoricalLoader
from polymarket_playground.execution.paper_executor import PaperExecutor
from polymarket_playground.markets.base_adapter import MarketAdapter


class TradingEnv(gym.Env):
    """
    Market-agnostic, agent-agnostic Gymnasium environment.

    Parameterized entirely by:
      - adapter: which MarketAdapter to use
      - config:  position limits, step size, episode filters
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        adapter: MarketAdapter,
        config: dict | None = None,
    ):
        super().__init__()
        self.adapter = adapter
        self.config = config or {}

        self.executor = PaperExecutor(config=self.config)
        max_steps = self.config.get("max_steps", 50)
        self.data = HistoricalLoader(adapter, max_steps=max_steps)
        self.position: Position | None = None
        self.state: MarketState | None = None

        # Observation + action spaces
        self.observation_space = self._build_obs_space()
        self.action_space = spaces.Dict(
            {
                "direction": spaces.Discrete(4),  # 0=hold, 1=yes, 2=no, 3=close
                "size": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            }
        )

    def _build_obs_space(self) -> spaces.Box:
        # 10 base features + variable signal features.
        # We call fetch_signals with dummy args just to discover how many
        # signal keys this adapter produces.  Guard against adapters that
        # cannot handle a ``None`` timestamp.
        try:
            n_signals = len(self.adapter.fetch_signals("", None))
        except Exception:
            n_signals = 0
        n_features = 10 + n_signals
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.data.next_episode()
        self.position = Position()
        return self._obs(), {}

    def step(self, action: Action | dict):
        if isinstance(action, dict):
            action = Action.from_dict(action)

        fill = self.executor.execute(action, self.state)
        self.position.update(fill)
        self.state = self.data.advance()

        reward = self._reward(fill)
        terminated = self.state.is_resolved
        truncated = self.data.is_truncated()

        if terminated:
            reward += self.position.settlement_pnl(self.state.resolution)

        return self._obs(), reward, terminated, truncated, self._info()

    def _obs(self) -> np.ndarray:
        """Flatten MarketState into a fixed-size float vector."""
        return ObservationBuilder.flatten(self.state)

    def _reward(self, fill: Fill | None) -> float:
        return (
            self.position.mtm_delta(self.state.yes_price)
            + (fill.spread_capture if fill else 0.0)
            - (fill.fees if fill else 0.0)
        )

    def _info(self) -> dict:
        return {
            "position": {
                "yes_shares": self.position.yes_shares,
                "no_shares": self.position.no_shares,
                "net_exposure": self.position.net_exposure,
            },
            "state": {
                "yes_price": self.state.yes_price,
                "time_elapsed": self.state.time_elapsed,
                "is_resolved": self.state.is_resolved,
            },
        }
