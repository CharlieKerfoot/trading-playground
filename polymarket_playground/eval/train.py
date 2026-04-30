"""RL training infrastructure using Stable-Baselines3."""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from polymarket_playground.core.action import Action
from polymarket_playground.core.env import TradingEnv

logger = logging.getLogger(__name__)


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
# Default hyperparameters tuned for trading environments
# ---------------------------------------------------------------------------

_PPO_DEFAULTS = {
    "learning_rate": 1e-4,
    "n_steps": 1024,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.995,
    "gae_lambda": 0.98,
    "clip_range": 0.2,
    "ent_coef": 0.05,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "normalize_advantage": True,
    "policy_kwargs": {
        "net_arch": dict(pi=[256, 256], vf=[256, 256]),
    },
}

_A2C_DEFAULTS = {
    "learning_rate": 3e-4,
    "n_steps": 128,
    "gamma": 0.995,
    "gae_lambda": 0.98,
    "ent_coef": 0.05,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "normalize_advantage": True,
    "policy_kwargs": {
        "net_arch": dict(pi=[256, 256], vf=[256, 256]),
    },
}

_ALGO_DEFAULTS = {
    "PPO": _PPO_DEFAULTS,
    "A2C": _A2C_DEFAULTS,
}


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
        constructor. Unspecified keys fall back to tuned defaults.
    """

    ALGORITHMS: dict[str, type] = {}  # populated lazily to avoid import cost

    def __init__(
        self,
        env: TradingEnv,
        algorithm: str = "PPO",
        config: dict[str, Any] | None = None,
    ) -> None:
        from stable_baselines3 import PPO, A2C
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        self.ALGORITHMS = {"PPO": PPO, "A2C": A2C}

        algorithm = algorithm.upper()
        if algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. "
                f"Choose from: {list(self.ALGORITHMS)}"
            )

        self.algorithm_name = algorithm
        self.wrapped_env = GymWrapper(env)

        # Wrap with VecNormalize for automatic observation and reward normalization
        vec_env = DummyVecEnv([lambda: self.wrapped_env])
        self.vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.995,
        )

        # Merge user config over tuned defaults
        defaults = _ALGO_DEFAULTS.get(algorithm, {}).copy()
        if config:
            # Deep-merge policy_kwargs if both exist
            if "policy_kwargs" in defaults and "policy_kwargs" in config:
                defaults["policy_kwargs"] = {
                    **defaults["policy_kwargs"],
                    **config.pop("policy_kwargs"),
                }
            defaults.update(config)
        self.config = defaults

        algo_cls = self.ALGORITHMS[algorithm]
        self.model = algo_cls(
            "MlpPolicy",
            self.vec_env,
            verbose=1,
            **self.config,
        )
        logger.info(
            "Initialized %s with config: %s",
            algorithm,
            {k: v for k, v in self.config.items() if k != "policy_kwargs"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, timesteps: int = 100_000) -> None:
        """Train the SB3 model for *timesteps* environment steps."""
        import copy

        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        # Create a separate eval env sharing normalization stats from training.
        # Deep-copy the underlying TradingEnv so eval episodes do not mutate
        # the training env's internal state (and vice versa).
        eval_wrapped = GymWrapper(copy.deepcopy(self.wrapped_env.env))
        eval_vec = DummyVecEnv([lambda: eval_wrapped])
        eval_vec = VecNormalize(
            eval_vec,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            gamma=0.995,
            training=False,
        )
        # Sync running stats so eval observations are normalized identically
        eval_vec.obs_rms = self.vec_env.obs_rms
        eval_vec.ret_rms = self.vec_env.ret_rms

        eval_callback = EvalCallback(
            eval_vec,
            best_model_save_path=None,
            log_path=None,
            eval_freq=max(1024, timesteps // 20),
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        )
        self.model.learn(total_timesteps=timesteps, callback=eval_callback)

    def save(self, path: str | pathlib.Path) -> None:
        """Save the trained model to *path* (SB3 native format)."""
        self.model.save(str(path))
        # Also save normalization stats so loaded models use same scaling
        norm_path = str(path).replace(".zip", "") + "_vecnorm.pkl"
        self.vec_env.save(norm_path)

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
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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

        vec_env = DummyVecEnv([lambda: trainer.wrapped_env])

        # Try to load normalization stats
        norm_path = str(path).replace(".zip", "") + "_vecnorm.pkl"
        try:
            trainer.vec_env = VecNormalize.load(norm_path, vec_env)
            trainer.vec_env.training = False
            trainer.vec_env.norm_reward = False
        except FileNotFoundError:
            trainer.vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
            trainer.vec_env.training = False

        trainer.config = {}
        trainer.model = algo_cls.load(str(path), env=trainer.vec_env)
        return trainer
