"""RL training infrastructure using Stable-Baselines3."""

from __future__ import annotations

import pathlib
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from polymarket_playground.core.action import Action
from polymarket_playground.core.env import TradingEnv


# ---------------------------------------------------------------------------
# Gym wrapper — converts TradingEnv's Dict action space into a
# MultiDiscrete space that SB3 algorithms (PPO, A2C) can handle.
# ---------------------------------------------------------------------------


class GymWrapper(gym.Env):
    """Wraps :class:`TradingEnv` for Stable-Baselines3 compatibility.

    SB3's on-policy algorithms (PPO, A2C) do **not** support
    ``spaces.Dict`` action spaces.  This wrapper re-exposes the
    environment with:

    * **action_space** = ``MultiDiscrete([4, 10])``
      - index 0: direction (0=hold, 1=buy_yes, 2=buy_no, 3=close)
      - index 1: size bucket 0-9, mapped linearly to 0.0-1.0
    * **observation_space** = same ``Box`` as the underlying env.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: TradingEnv) -> None:
        super().__init__()
        self.env = env

        # Mirror the observation space from the inner env
        self.observation_space = env.observation_space

        # MultiDiscrete: [direction(4), size_bucket(10)]
        self.action_space = spaces.MultiDiscrete([4, 10])

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action: np.ndarray):
        direction = int(action[0])
        size = int(action[1]) / 9.0  # bucket 0-9 -> 0.0-1.0
        trading_action = Action(direction=direction, size=size, reasoning="rl_policy")
        return self.env.step(trading_action)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class RLTrainer:
    """High-level wrapper around SB3 training.

    Usage::

        from polymarket_playground.core.env import TradingEnv
        from polymarket_playground.data.market_cache import MarketCache

        env = TradingEnv(cache=MarketCache())
        trainer = RLTrainer(env, algorithm="PPO")
        trainer.train(timesteps=100_000)
        trainer.save("models/btc_ppo")

    Parameters
    ----------
    env:
        A :class:`TradingEnv` instance.  It will automatically be wrapped
        with :class:`GymWrapper` for SB3 compatibility.
    algorithm:
        ``"PPO"`` or ``"A2C"`` (case-insensitive).
    config:
        Optional dict of keyword arguments forwarded to the SB3 algorithm
        constructor (e.g. ``{"learning_rate": 3e-4, "n_steps": 2048}``).
    """

    ALGORITHMS: dict[str, type] = {}  # populated lazily to avoid import cost

    def __init__(
        self,
        env: TradingEnv,
        algorithm: str = "PPO",
        config: dict[str, Any] | None = None,
    ) -> None:
        from stable_baselines3 import PPO, A2C

        self.ALGORITHMS = {"PPO": PPO, "A2C": A2C}

        algorithm = algorithm.upper()
        if algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. "
                f"Choose from: {list(self.ALGORITHMS)}"
            )

        self.algorithm_name = algorithm
        self.wrapped_env = GymWrapper(env)
        self.config = config or {}

        algo_cls = self.ALGORITHMS[algorithm]
        self.model = algo_cls(
            "MlpPolicy",
            self.wrapped_env,
            verbose=1,
            **self.config,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, timesteps: int = 100_000) -> None:
        """Train the SB3 model for *timesteps* environment steps."""
        self.model.learn(total_timesteps=timesteps)

    def save(self, path: str | pathlib.Path) -> None:
        """Save the trained model to *path* (SB3 native format)."""
        self.model.save(str(path))

    @classmethod
    def load(
        cls,
        path: str | pathlib.Path,
        env: TradingEnv,
        algorithm: str = "PPO",
    ) -> RLTrainer:
        """Load a previously saved model and return an ``RLTrainer``.

        Parameters
        ----------
        path:
            Path to the saved model (without or with ``.zip``).
        env:
            A :class:`TradingEnv` to attach to the loaded model.
        algorithm:
            The algorithm that was used to train the model.
        """
        from stable_baselines3 import PPO, A2C

        algo_map = {"PPO": PPO, "A2C": A2C}
        algorithm = algorithm.upper()
        algo_cls = algo_map.get(algorithm)
        if algo_cls is None:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. Choose from: {list(algo_map)}"
            )

        trainer = object.__new__(cls)
        trainer.algorithm_name = algorithm
        trainer.wrapped_env = GymWrapper(env)
        trainer.config = {}
        trainer.model = algo_cls.load(str(path), env=trainer.wrapped_env)
        return trainer
