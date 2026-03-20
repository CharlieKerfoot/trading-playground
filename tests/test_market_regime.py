"""Tests for market regime classification."""

from polymarket_playground.data.market_cache import MarketCache


def test_classify_neutral_short():
    """Short price history returns neutral."""
    assert MarketCache.classify_regime([0.5] * 5) == "neutral"


def test_classify_neutral_flat():
    """Flat prices return neutral."""
    assert MarketCache.classify_regime([0.5] * 20) == "neutral"


def test_classify_trending():
    """Exponentially increasing prices should be trending (positive autocorrelation)."""
    prices = [0.3 + 0.3 * (i / 59) ** 2 for i in range(60)]
    regime = MarketCache.classify_regime(prices)
    assert regime == "trending"


def test_classify_low_liquidity():
    """Low volume with low volatility should be tagged as low liquidity."""
    # Needs vol < 0.05 to not trigger high_volatility first
    prices = [0.50 + 0.001 * (i % 5) for i in range(60)]
    regime = MarketCache.classify_regime(prices, volume=500)
    assert regime == "low_liquidity"


def test_classify_high_volatility():
    """Highly volatile prices should be tagged as high_volatility."""
    import math
    prices = [0.5 + 0.15 * math.sin(i * 1.0) for i in range(60)]
    regime = MarketCache.classify_regime(prices)
    assert regime == "high_volatility"


def test_tag_regimes(test_cache):
    """tag_regimes should tag all untagged markets."""
    tagged = test_cache.tag_regimes()
    assert tagged == 1

    # Second call should tag 0 (already tagged)
    tagged2 = test_cache.tag_regimes()
    assert tagged2 == 0


def test_get_market_ids_by_regime(test_cache):
    """After tagging, should be able to filter by regime."""
    test_cache.tag_regimes()

    # Get all regimes and check our market is in one
    for regime in ["trending", "mean_reverting", "high_volatility", "low_liquidity", "neutral"]:
        ids = test_cache.get_market_ids_by_regime(regime)
        if ids:
            assert "test-market-001" in ids
            return

    # Market should be in at least one regime
    assert False, "Market was not found in any regime"


def test_stats_includes_regimes(test_cache):
    """stats() should include regime breakdown."""
    test_cache.tag_regimes()
    stats = test_cache.stats()
    assert "regimes" in stats
    assert len(stats["regimes"]) > 0
