"""Tests for HistoricalLoader episode generation and resampling."""

from polymarket_playground.data.historical_loader import HistoricalLoader


def test_episode_generation(test_cache):
    """Loader produces valid episodes from cached data."""
    loader = HistoricalLoader(cache=test_cache, max_steps=10, seed=42)

    assert loader.available_markets == 1

    state = loader.next_episode()
    assert state.market_id == "test-market-001"
    assert state.question == "Will it rain tomorrow?"
    assert not state.is_resolved
    assert 0.0 < state.yes_price < 1.0

    # Advance through episode
    for i in range(9):
        state = loader.advance()
        if i < 8:
            assert not state.is_resolved
        else:
            # Last step should be resolved
            assert state.is_resolved
            assert state.resolution == 1.0
            # Terminal price should be snapped to resolution
            assert state.yes_price == 1.0
            assert state.no_price == 0.0


def test_resampling_short_history(tmp_path):
    """Loader resamples when price history is shorter than max_steps."""
    import json
    from polymarket_playground.data.market_cache import MarketCache

    db_path = tmp_path / "short.db"
    cache = MarketCache(db_path=db_path)

    market_id = "short-market"
    meta = {"question": "Short?", "liquidityNum": 1000, "volume24hr": 100}

    with cache._lock:
        cache._conn.execute(
            "INSERT INTO markets (id, question, outcome, volume, metadata, category) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (market_id, "Short?", 1.0, 100.0, json.dumps(meta), "Test"),
        )
        # Only 5 price points (much less than max_steps=20)
        for i, p in enumerate([0.3, 0.4, 0.5, 0.6, 0.7]):
            cache._conn.execute(
                "INSERT INTO price_history (market_id, timestamp, price) VALUES (?, ?, ?)",
                (market_id, 1000 + i * 60, p),
            )
        cache._conn.commit()

    loader = HistoricalLoader(cache=cache, max_steps=20, seed=42)
    state = loader.next_episode()

    # Should not crash — resampling fills the window
    assert state.market_id == market_id

    # Should be able to advance through all 20 steps
    for _ in range(19):
        state = loader.advance()

    assert state.is_resolved
    cache.close()


def test_no_markets_raises(tmp_path):
    """Loader raises RuntimeError when cache is empty."""
    import pytest
    from polymarket_playground.data.market_cache import MarketCache

    db_path = tmp_path / "empty.db"
    cache = MarketCache(db_path=db_path)
    loader = HistoricalLoader(cache=cache, max_steps=10)

    with pytest.raises(RuntimeError, match="No markets in cache"):
        loader.next_episode()

    cache.close()
