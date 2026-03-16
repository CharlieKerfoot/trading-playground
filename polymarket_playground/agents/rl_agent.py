"""RL policy wrapper (SB3-compatible)."""

from __future__ import annotations

import pathlib
import random

import numpy as np

from polymarket_playground.core.action import Action
from polymarket_playground.core.state import MarketState, ObservationBuilder
from polymarket_playground.core.types import EpisodeResult

from .base_agent import BaseAgent


class RLAgent(BaseAgent):
    """Wraps a Stable-Baselines3-compatible policy as a ``BaseAgent``.

    If no policy is provided the agent acts randomly, which is useful as a
    baseline or for smoke-testing the environment loop.
    """

    def __init__(self, policy: object | None = None) -> None:
        self.policy = policy
        self._yes_shares: float = 0.0
        self._no_shares: float = 0.0

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, state: MarketState) -> Action:
        obs = self._build_obs(state)

        if self.policy is None:
            direction = random.randint(0, 2)  # hold / buy_yes / buy_no
            size = random.random()
            action = Action(direction=direction, size=size, reasoning="random")
        else:
            raw_action, _ = self.policy.predict(obs, deterministic=True)
            if hasattr(raw_action, "__getitem__"):
                direction = int(raw_action[0]) if len(raw_action) > 0 else 0
                size_bucket = int(raw_action[1]) if len(raw_action) > 1 else 5
                size = size_bucket / 9.0
            else:
                direction = int(raw_action)
                size = 0.5
            action = Action(direction=direction, size=size, reasoning="rl_policy")

        # Track position locally
        if action.direction == 1:
            self._yes_shares += action.size
        elif action.direction == 2:
            self._no_shares += action.size
        elif action.direction == 3:
            self._yes_shares = 0.0
            self._no_shares = 0.0

        return action

    def _build_obs(self, state: MarketState) -> np.ndarray:
        """Build observation matching TradingEnv's obs space (market + position)."""
        market_obs = ObservationBuilder.flatten(state)
        position_obs = np.array([
            self._yes_shares,
            self._no_shares,
            self._yes_shares - self._no_shares,
        ], dtype=np.float32)
        return np.concatenate([market_obs, position_obs])

    def on_reset(self, state: MarketState) -> None:
        self._yes_shares = 0.0
        self._no_shares = 0.0

    def on_episode_end(self, result: EpisodeResult) -> None:
        pass

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> None:
        """Save the trained SB3 model to *path*.

        The SB3 model is saved using its native ``save()`` method.  The
        resulting file can be loaded back with :meth:`RLAgent.load`.
        """
        if self.policy is None:
            raise ValueError("Cannot save: no policy has been set.")
        self.policy.save(str(path))

    @classmethod
    def load(cls, path: str | pathlib.Path, algorithm: str = "PPO") -> RLAgent:
        """Load a trained SB3 model from *path* and return an ``RLAgent``.

        Parameters
        ----------
        path:
            File path (without or with ``.zip`` extension) to load from.
        algorithm:
            The SB3 algorithm class name used during training (``"PPO"``
            or ``"A2C"``).
        """
        from stable_baselines3 import PPO, A2C

        algo_map = {"PPO": PPO, "A2C": A2C}
        algo_cls = algo_map.get(algorithm.upper())
        if algo_cls is None:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. Choose from: {list(algo_map)}"
            )

        model = algo_cls.load(str(path))
        return cls(policy=model)
