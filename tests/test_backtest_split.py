"""Tests for walk-forward backtesting (date-based market splits)."""

import tempfile
from pathlib import Path

import pytest

from polymarket_playground.data.market_cache import MarketCache


@pytest.fixture
def split_cache():
    """Create a cache with markets at different timestamps for split testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    cache = MarketCache(db_path=db_path)

    # Insert markets with different time ranges
    conn = cache._conn

    # Market A: old (timestamps 1000-2000)
    conn.execute(
        "INSERT INTO markets (id, question, outcome, volume) VALUES (?, ?, ?, ?)",
        ("market_a", "Old market?", 1.0, 100),
    )
    for t in range(1000, 2001, 100):
        conn.execute(
            "INSERT INTO price_history (market_id, timestamp, price) VALUES (?, ?, ?)",
            ("market_a", t, 0.5),
        )

    # Market B: middle (timestamps 1500-3000)
    conn.execute(
        "INSERT INTO markets (id, question, outcome, volume) VALUES (?, ?, ?, ?)",
        ("market_b", "Middle market?", 0.0, 200),
    )
    for t in range(1500, 3001, 100):
        conn.execute(
            "INSERT INTO price_history (market_id, timestamp, price) VALUES (?, ?, ?)",
            ("market_b", t, 0.6),
        )

    # Market C: new (timestamps 2500-4000)
    conn.execute(
        "INSERT INTO markets (id, question, outcome, volume) VALUES (?, ?, ?, ?)",
        ("market_c", "New market?", 1.0, 300),
    )
    for t in range(2500, 4001, 100):
        conn.execute(
            "INSERT INTO price_history (market_id, timestamp, price) VALUES (?, ?, ?)",
            ("market_c", t, 0.7),
        )

    # Market D: no outcome (should be excluded)
    conn.execute(
        "INSERT INTO markets (id, question, outcome, volume) VALUES (?, ?, ?, ?)",
        ("market_d", "No outcome?", None, 50),
    )
    for t in range(1000, 2001, 100):
        conn.execute(
            "INSERT INTO price_history (market_id, timestamp, price) VALUES (?, ?, ?)",
            ("market_d", t, 0.4),
        )

    conn.commit()
    yield cache
    cache.close()
    Path(db_path).unlink(missing_ok=True)


def test_get_earliest_latest(split_cache):
    earliest, latest = split_cache.get_earliest_latest_timestamps()
    assert earliest == 1000
    assert latest == 4000


def test_get_markets_before_cutoff(split_cache):
    """Markets with prices before timestamp 2000."""
    ids = split_cache.get_market_ids_by_date_range(before=2000)
    assert "market_a" in ids
    assert "market_b" in ids
    # market_d excluded because outcome is None
    assert "market_d" not in ids


def test_get_markets_after_cutoff(split_cache):
    """Markets with prices at or after timestamp 3000."""
    ids = split_cache.get_market_ids_by_date_range(after=3000)
    assert "market_b" in ids  # has prices up to 3000
    assert "market_c" in ids
    assert "market_a" not in ids  # latest price at 2000


def test_train_test_split(split_cache):
    """Splitting at 2500 should separate old from new markets."""
    train = split_cache.get_market_ids_by_date_range(before=2500)
    test = split_cache.get_market_ids_by_date_range(after=2500)

    assert "market_a" in train
    assert "market_c" in test
    # market_b spans the cutoff, appears in both
    assert "market_b" in train
    assert "market_b" in test


def test_empty_range(split_cache):
    """Timestamp range with no data returns empty list."""
    ids = split_cache.get_market_ids_by_date_range(after=99999)
    assert ids == []


def test_set_market_ids_on_loader(split_cache):
    """HistoricalLoader.set_market_ids restricts the pool."""
    from polymarket_playground.data.historical_loader import HistoricalLoader

    loader = HistoricalLoader(split_cache, max_steps=5)
    original_count = loader.available_markets

    loader.set_market_ids(["market_a"])
    assert loader.available_markets == 1

    loader.set_market_ids(["market_a", "market_c"])
    assert loader.available_markets == 2
