"""Shared fixtures for the test suite."""

import json
import sqlite3

import pytest

from polymarket_playground.data.market_cache import MarketCache


@pytest.fixture
def test_cache(tmp_path):
    """Create a MarketCache with a seeded test database.

    Contains one resolved market (YES wins, outcome=1.0) with 60 price
    points trending from 0.4 to 0.8, giving meaningful training signal.
    """
    db_path = tmp_path / "test.db"
    cache = MarketCache(db_path=db_path)

    market_id = "test-market-001"
    meta = {
        "question": "Will it rain tomorrow?",
        "liquidityNum": 5000,
        "volume24hr": 10000,
        "groupItemTitle": "Weather",
    }

    with cache._lock:
        cache._conn.execute(
            "INSERT INTO markets (id, question, outcome, volume, metadata, category) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (market_id, meta["question"], 1.0, 10000.0, json.dumps(meta), "Weather"),
        )

        # 60 price points trending from 0.4 to 0.8
        prices = [0.4 + 0.4 * (i / 59) for i in range(60)]
        cache._conn.executemany(
            "INSERT INTO price_history (market_id, timestamp, price) VALUES (?, ?, ?)",
            [(market_id, 1000 + i * 60, p) for i, p in enumerate(prices)],
        )
        cache._conn.commit()

    yield cache
    cache.close()


@pytest.fixture
def test_cache_no_win(tmp_path):
    """MarketCache with a NO-wins market (outcome=0.0)."""
    db_path = tmp_path / "test_no.db"
    cache = MarketCache(db_path=db_path)

    market_id = "test-market-no"
    meta = {"question": "Will it snow?", "liquidityNum": 5000, "volume24hr": 5000}

    with cache._lock:
        cache._conn.execute(
            "INSERT INTO markets (id, question, outcome, volume, metadata, category) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (market_id, meta["question"], 0.0, 5000.0, json.dumps(meta), "Weather"),
        )

        # 60 price points trending from 0.6 to 0.3 (market was wrong)
        prices = [0.6 - 0.3 * (i / 59) for i in range(60)]
        cache._conn.executemany(
            "INSERT INTO price_history (market_id, timestamp, price) VALUES (?, ?, ?)",
            [(market_id, 1000 + i * 60, p) for i, p in enumerate(prices)],
        )
        cache._conn.commit()

    yield cache
    cache.close()
