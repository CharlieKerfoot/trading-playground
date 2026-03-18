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
from polymarket_playground.data.market_cache import MarketCache
from polymarket_playground.execution.paper_executor import PaperExecutor


class TradingEnv(gym.Env):
    """
    Market-agnostic, agent-agnostic Gymnasium environment.

    Parameterized entirely by:
      - cache: MarketCache with pre-synced market data
      - config: position limits, step size, episode filters
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        cache: MarketCache,
        config: dict | None = None,
        signal_provider: object | None = None,
    ):
        super().__init__()
        self.cache = cache
        self.config = config or {}

        self.executor = PaperExecutor(config=self.config)
        max_steps = self.config.get("max_steps", 50)
        self.position_limit: float = self.config.get("position_limit", 10.0)
        self.data = HistoricalLoader(cache, max_steps=max_steps, signal_provider=signal_provider)
        self.position: Position | None = None
        self.state: MarketState | None = None
        self._total_fees: float = 0.0

        # Observation: 10 base + 8 signals + 3 position features
        n_features = 10 + 8 + 3
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(n_features,), dtype=np.float32
        )
        self.action_space = spaces.Dict(
            {
                "direction": spaces.Discrete(4),  # 0=hold, 1=yes, 2=no, 3=close
                "size": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            }
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.data.next_episode()
        self.position = Position()
        self._total_fees = 0.0
        self._sync_position_to_state()
        return self._obs(), {}

    def step(self, action: Action | dict):
        if isinstance(action, dict):
            action = Action.from_dict(action)

        # Handle close on flat position as hold
        if action.direction == 3:  # close
            if self.position.is_flat:
                action = Action(direction=0, size=0.0)  # treat as hold
            else:
                # Override size to actual position size
                actual_size = self.position.yes_shares + self.position.no_shares
                action = Action(direction=3, size=actual_size, reasoning=action.reasoning)

        # Clamp action size so position doesn't exceed limit
        if action.direction in (1, 2):  # buy_yes or buy_no
            current_shares = (
                self.position.yes_shares if action.direction == 1
                else self.position.no_shares
            )
            headroom = max(0.0, self.position_limit - current_shares)
            action = Action(
                direction=action.direction,
                size=min(action.size, headroom),
                reasoning=action.reasoning,
            )
            if action.size <= 0:
                action = Action(direction=0, size=0.0)  # treat as hold

        fill = self.executor.execute(action, self.state, self.position)
        self.position.update(fill)
        self.state = self.data.advance()
        self._sync_position_to_state()

        # Track cumulative fees for info
        if fill:
            self._total_fees += fill.fees

        reward = self._reward(fill)
        terminated = self.state.is_resolved
        truncated = not terminated and self.data.is_truncated()

        if terminated:
            # Settlement reward: the full P&L from resolution.
            # settlement_pnl accounts for cost basis and gives the actual
            # profit/loss from holding through to resolution.
            reward += self.position.settlement_pnl(self.state.resolution) - (
                self.position.close_value(self.state.yes_price)
                - self.position.cost_basis
            )

        return self._obs(), reward, terminated, truncated, self._info(fill)

    def _sync_position_to_state(self) -> None:
        """Copy actual position into MarketState so agents can see it."""
        self.state.agent_yes_shares = self.position.yes_shares
        self.state.agent_no_shares = self.position.no_shares
        self.state.agent_cost_basis = self.position.cost_basis
        self.state.agent_realized_pnl = self.position.realized_pnl

    def _obs(self) -> np.ndarray:
        """Flatten MarketState + position into a fixed-size float vector."""
        market_obs = ObservationBuilder.flatten(self.state)
        # Normalize position features to ~[0, 1] range
        pos_limit = self.position_limit
        position_obs = np.array([
            self.position.yes_shares / pos_limit,
            self.position.no_shares / pos_limit,
            self.position.net_exposure / pos_limit,
        ], dtype=np.float32)
        return np.concatenate([market_obs, position_obs])

    def _reward(self, fill: Fill | None) -> float:
        return (
            self.position.mtm_delta(self.state.yes_price)
            + (fill.spread_capture if fill else 0.0)
            - (fill.fees if fill else 0.0)
        )

    def _info(self, fill: Fill | None = None) -> dict:
        return {
            "position": {
                "yes_shares": self.position.yes_shares,
                "no_shares": self.position.no_shares,
                "net_exposure": self.position.net_exposure,
                "cost_basis": self.position.cost_basis,
                "realized_pnl": self.position.realized_pnl,
            },
            "state": {
                "yes_price": self.state.yes_price,
                "time_elapsed": self.state.time_elapsed,
                "is_resolved": self.state.is_resolved,
            },
            "fill": {
                "direction": fill.direction if fill else None,
                "size": fill.size if fill else 0.0,
                "price": fill.price if fill else 0.0,
                "fees": fill.fees if fill else 0.0,
            },
            "spread_capture": fill.spread_capture if fill else 0.0,
        }
