"""Tests for statistical validation and walk-forward backtesting."""

import math

from polymarket_playground.core.types import EpisodeResult
from polymarket_playground.eval.metrics import (
    ValidationResult,
    bootstrap_sharpe_ci,
    compare_vs_random,
    t_test_returns,
    validate_strategy,
)


def _make_results(rewards: list[float]) -> list[EpisodeResult]:
    """Helper: create EpisodeResult list from raw reward values."""
    results = []
    for r in rewards:
        er = EpisodeResult()
        er.total_reward = r
        er.rewards = [r]
        er.n_steps = 1
        results.append(er)
    return results


def test_t_test_positive_returns():
    """Positive returns should give positive t-stat and low p-value."""
    results = _make_results([0.1, 0.2, 0.15, 0.12, 0.18] * 10)
    t_stat, p_val = t_test_returns(results)
    assert t_stat > 0
    assert p_val < 0.01


def test_t_test_zero_returns():
    """Zero-mean returns should give high p-value."""
    results = _make_results([0.1, -0.1, 0.05, -0.05, 0.0] * 10)
    t_stat, p_val = t_test_returns(results)
    assert p_val > 0.3


def test_t_test_too_few():
    """With < 2 results, return default values."""
    results = _make_results([0.1])
    t_stat, p_val = t_test_returns(results)
    assert t_stat == 0.0
    assert p_val == 1.0


def test_t_test_zero_variance():
    """All identical returns should give p=1.0."""
    results = _make_results([0.5] * 20)
    t_stat, p_val = t_test_returns(results)
    assert p_val == 1.0


def test_bootstrap_sharpe_ci():
    """CI should bracket the point estimate."""
    results = _make_results([0.1, 0.2, 0.05, 0.15, 0.12] * 10)
    lo, hi = bootstrap_sharpe_ci(results, n_bootstrap=5000)
    assert lo < hi
    assert lo > -5.0  # sanity check
    assert hi < 10.0


def test_bootstrap_too_few():
    results = _make_results([0.1])
    lo, hi = bootstrap_sharpe_ci(results)
    assert lo == 0.0
    assert hi == 0.0


def test_validate_strategy_significant():
    """Strategy with consistent positive returns should be significant."""
    results = _make_results([0.1, 0.2, 0.15, 0.12, 0.18, 0.09, 0.14] * 5)
    v = validate_strategy(results, min_episodes=30)
    assert v.status == "significant"
    assert v.is_significant is True
    assert v.p_value < 0.05
    assert v.mean_return > 0


def test_validate_strategy_insufficient():
    """Too few episodes should return insufficient_data."""
    results = _make_results([0.1, 0.2, 0.3])
    v = validate_strategy(results, min_episodes=30)
    assert v.status == "insufficient_data"
    assert v.is_significant is False


def test_validate_strategy_not_significant():
    """Zero-mean strategy should not be significant."""
    results = _make_results([0.1, -0.1, 0.05, -0.05, 0.02, -0.02] * 6)
    v = validate_strategy(results, min_episodes=30)
    assert v.status == "not_significant"
    assert v.is_significant is False


def test_compare_vs_random_better():
    """Agent with consistently better returns should beat random."""
    agent = _make_results([0.15, 0.2, 0.18, 0.12, 0.14] * 10)
    rand = _make_results([-0.02, 0.01, -0.03, 0.02, -0.01] * 10)
    is_better, p_val = compare_vs_random(agent, rand)
    assert is_better is True
    assert p_val < 0.05


def test_compare_vs_random_same():
    """Similar distributions should not show significance."""
    agent = _make_results([0.05, -0.05, 0.03, -0.03, 0.01] * 10)
    rand = _make_results([0.04, -0.04, 0.02, -0.02, 0.01] * 10)
    is_better, p_val = compare_vs_random(agent, rand)
    assert p_val > 0.05


def test_compare_vs_random_too_few():
    agent = _make_results([0.1])
    rand = _make_results([0.05])
    is_better, p_val = compare_vs_random(agent, rand)
    assert is_better is False
    assert p_val == 1.0


def test_validation_result_str():
    """Verify string representation works."""
    v = ValidationResult(
        is_significant=True,
        p_value=0.01,
        t_statistic=3.5,
        mean_return=0.05,
        std_return=0.02,
        n_episodes=100,
        sharpe=2.5,
        sharpe_ci_lower=1.8,
        sharpe_ci_upper=3.2,
        min_episodes_needed=30,
        status="significant",
    )
    s = str(v)
    assert "YES" in s
    assert "0.0100" in s
