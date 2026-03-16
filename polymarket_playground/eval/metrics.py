"""Performance metrics computed from episode results."""

from __future__ import annotations

import math

from polymarket_playground.core.types import EpisodeResult


def sharpe_ratio(results: list[EpisodeResult]) -> float:
    """Reward-based Sharpe ratio across episodes."""
    if len(results) < 2:
        return 0.0
    rewards = [r.total_reward for r in results]
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1)
    std = math.sqrt(variance)
    if std == 0.0:
        return 0.0
    return mean / std


def win_rate(results: list[EpisodeResult]) -> float:
    """Fraction of episodes with positive total reward."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.total_reward > 0) / len(results)


def avg_pnl(results: list[EpisodeResult]) -> float:
    """Mean total reward across episodes."""
    if not results:
        return 0.0
    return sum(r.total_reward for r in results) / len(results)


def max_drawdown(results: list[EpisodeResult]) -> float:
    """Worst cumulative drawdown within any single episode.

    Returns a non-positive value (0.0 means no drawdown occurred).
    """
    worst = 0.0
    for result in results:
        cumulative = 0.0
        peak = 0.0
        for reward in result.rewards:
            cumulative += reward
            if cumulative > peak:
                peak = cumulative
            drawdown = cumulative - peak
            if drawdown < worst:
                worst = drawdown
    return worst


def spread_capture(results: list[EpisodeResult]) -> float:
    """Total spread captured from fills across all episodes.

    Looks for a ``spread_capture`` key in each step's info dict.
    """
    total = 0.0
    for result in results:
        for info in result.infos:
            total += info.get("spread_capture", 0.0)
    return total
