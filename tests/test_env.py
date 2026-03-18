"""Tests for TradingEnv lifecycle, terminal reward, and action clamping."""

from polymarket_playground.core.action import Action
from polymarket_playground.core.env import TradingEnv


def test_env_lifecycle(test_cache):
    """Full episode: reset -> step loop -> terminated on resolution."""
    env = TradingEnv(cache=test_cache, config={"max_steps": 10})
    obs, info = env.reset(seed=42)

    assert obs.shape == (21,)
    assert env.state is not None
    assert env.position.is_flat

    terminated = False
    truncated = False
    steps = 0
    while not terminated and not truncated:
        obs, reward, terminated, truncated, info = env.step(
            Action(direction=0, size=0.0)  # hold every step
        )
        steps += 1

    # Episode resolves on step max_steps-1 (0-indexed internally)
    assert steps == 9
    assert terminated


def test_terminal_reward_yes_wins(test_cache):
    """Agent holding YES shares when YES wins should get positive terminal reward."""
    env = TradingEnv(cache=test_cache, config={"max_steps": 10})
    env.reset(seed=42)

    # Buy YES shares on first step
    obs, reward, terminated, truncated, info = env.step(
        Action(direction=1, size=5.0)
    )
    assert env.position.yes_shares == 5.0

    # Hold until terminal
    total_reward = reward
    while not terminated and not truncated:
        obs, reward, terminated, truncated, info = env.step(
            Action(direction=0, size=0.0)
        )
        total_reward += reward

    assert terminated
    # Market resolves YES (outcome=1.0). Agent bought YES at ~ask price
    # (around 0.4-0.5 range), so settlement at 1.0 should be profitable.
    assert total_reward > 0, f"Expected positive total reward, got {total_reward}"

    # Verify the state has resolution prices snapped
    assert env.state.is_resolved
    assert env.state.resolution == 1.0
    assert env.state.yes_price == 1.0


def test_terminal_reward_yes_shares_no_wins(test_cache_no_win):
    """Agent holding YES shares when NO wins should get negative terminal reward."""
    env = TradingEnv(cache=test_cache_no_win, config={"max_steps": 10})
    env.reset(seed=42)

    # Buy YES shares
    obs, reward, terminated, truncated, info = env.step(
        Action(direction=1, size=5.0)
    )

    # Hold until terminal
    total_reward = reward
    while not terminated and not truncated:
        obs, reward, terminated, truncated, info = env.step(
            Action(direction=0, size=0.0)
        )
        total_reward += reward

    assert terminated
    # Market resolves NO (outcome=0.0). YES shares become worthless.
    assert total_reward < 0, f"Expected negative total reward, got {total_reward}"
    assert env.state.yes_price == 0.0


def test_action_clamping_at_position_limit(test_cache):
    """Buy action at position limit should be clamped to hold."""
    env = TradingEnv(cache=test_cache, config={"max_steps": 10, "position_limit": 3.0})
    env.reset(seed=42)

    # Buy up to limit
    env.step(Action(direction=1, size=3.0))
    assert env.position.yes_shares == 3.0

    # Try to buy more — should become hold (no additional shares)
    obs, reward, terminated, truncated, info = env.step(
        Action(direction=1, size=5.0)
    )
    assert env.position.yes_shares == 3.0  # unchanged
    assert info["fill"]["direction"] is None  # no fill


def test_close_on_flat_becomes_hold(test_cache):
    """Close action on a flat position should become hold."""
    env = TradingEnv(cache=test_cache, config={"max_steps": 10})
    env.reset(seed=42)

    # Close with no position — should not crash
    obs, reward, terminated, truncated, info = env.step(
        Action(direction=3, size=1.0)
    )
    assert env.position.is_flat
    assert info["fill"]["direction"] is None  # treated as hold
