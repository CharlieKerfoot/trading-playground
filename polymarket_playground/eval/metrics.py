"""Performance metrics and statistical validation for episode results."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

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


# ---------------------------------------------------------------------------
# Statistical validation
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of statistical validation of a strategy."""

    is_significant: bool
    p_value: float
    t_statistic: float
    mean_return: float
    std_return: float
    n_episodes: int
    sharpe: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float
    min_episodes_needed: int
    status: str  # "significant", "not_significant", "insufficient_data"

    def __str__(self) -> str:
        if self.status == "insufficient_data":
            return (
                f"INSUFFICIENT DATA: {self.n_episodes} episodes "
                f"(need ≥{self.min_episodes_needed})"
            )
        sig = "YES" if self.is_significant else "NO"
        return (
            f"Significant: {sig} (p={self.p_value:.4f})\n"
            f"Mean return: {self.mean_return:+.6f} ± {self.std_return:.6f}\n"
            f"Sharpe: {self.sharpe:.4f} [{self.sharpe_ci_lower:.4f}, {self.sharpe_ci_upper:.4f}]\n"
            f"Episodes: {self.n_episodes}"
        )


def t_test_returns(results: list[EpisodeResult]) -> tuple[float, float]:
    """One-sample t-test of episode returns against zero.

    Returns (t_statistic, p_value). Tests whether mean return is
    significantly different from zero.
    """
    n = len(results)
    if n < 2:
        return 0.0, 1.0

    rewards = [r.total_reward for r in results]
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / (n - 1)
    std = math.sqrt(variance)

    if std == 0.0:
        return 0.0, 1.0

    t_stat = mean / (std / math.sqrt(n))

    # Two-tailed p-value approximation using the normal distribution
    # (good approximation for n > 30, conservative for smaller n)
    p_value = 2.0 * _normal_cdf(-abs(t_stat))
    return t_stat, p_value


def bootstrap_sharpe_ci(
    results: list[EpisodeResult],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the Sharpe ratio.

    Returns (lower_bound, upper_bound) at the given confidence level.
    """
    n = len(results)
    if n < 2:
        return 0.0, 0.0

    rng = random.Random(seed)
    rewards = [r.total_reward for r in results]
    sharpes: list[float] = []

    for _ in range(n_bootstrap):
        sample = [rng.choice(rewards) for _ in range(n)]
        mean = sum(sample) / n
        variance = sum((r - mean) ** 2 for r in sample) / (n - 1)
        std = math.sqrt(variance)
        if std > 0.0:
            sharpes.append(mean / std)
        else:
            sharpes.append(0.0)

    sharpes.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = max(0, int(alpha * len(sharpes)))
    hi_idx = min(len(sharpes) - 1, int((1.0 - alpha) * len(sharpes)))
    return sharpes[lo_idx], sharpes[hi_idx]


def validate_strategy(
    results: list[EpisodeResult],
    significance_level: float = 0.05,
    min_episodes: int = 30,
) -> ValidationResult:
    """Full statistical validation of a trading strategy.

    Checks:
    1. Sufficient sample size (≥ min_episodes)
    2. t-test: mean return significantly different from zero
    3. Bootstrap CI on Sharpe ratio
    """
    n = len(results)
    if n < min_episodes:
        return ValidationResult(
            is_significant=False,
            p_value=1.0,
            t_statistic=0.0,
            mean_return=avg_pnl(results) if results else 0.0,
            std_return=0.0,
            n_episodes=n,
            sharpe=sharpe_ratio(results) if results else 0.0,
            sharpe_ci_lower=0.0,
            sharpe_ci_upper=0.0,
            min_episodes_needed=min_episodes,
            status="insufficient_data",
        )

    t_stat, p_val = t_test_returns(results)
    sharpe = sharpe_ratio(results)
    ci_lo, ci_hi = bootstrap_sharpe_ci(results)
    mean = avg_pnl(results)

    rewards = [r.total_reward for r in results]
    variance = sum((r - mean) ** 2 for r in rewards) / (n - 1)
    std = math.sqrt(variance)

    is_sig = p_val < significance_level and mean > 0

    return ValidationResult(
        is_significant=is_sig,
        p_value=p_val,
        t_statistic=t_stat,
        mean_return=mean,
        std_return=std,
        n_episodes=n,
        sharpe=sharpe,
        sharpe_ci_lower=ci_lo,
        sharpe_ci_upper=ci_hi,
        min_episodes_needed=min_episodes,
        status="significant" if is_sig else "not_significant",
    )


def compare_vs_random(
    agent_results: list[EpisodeResult],
    random_results: list[EpisodeResult],
    significance_level: float = 0.05,
) -> tuple[bool, float]:
    """Two-sample t-test: does the agent beat a random baseline?

    Returns (is_better, p_value).
    """
    n1 = len(agent_results)
    n2 = len(random_results)
    if n1 < 2 or n2 < 2:
        return False, 1.0

    mean1 = avg_pnl(agent_results)
    mean2 = avg_pnl(random_results)

    rewards1 = [r.total_reward for r in agent_results]
    rewards2 = [r.total_reward for r in random_results]

    var1 = sum((r - mean1) ** 2 for r in rewards1) / (n1 - 1)
    var2 = sum((r - mean2) ** 2 for r in rewards2) / (n2 - 1)

    se = math.sqrt(var1 / n1 + var2 / n2)
    if se == 0.0:
        return False, 1.0

    t_stat = (mean1 - mean2) / se
    p_value = 2.0 * _normal_cdf(-abs(t_stat))
    return (mean1 > mean2 and p_value < significance_level), p_value


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    return 0.5 * math.erfc(-x / math.sqrt(2))
