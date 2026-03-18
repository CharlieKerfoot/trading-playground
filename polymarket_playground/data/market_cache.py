"""MarketCache — SQLite-backed cache for resolved Polymarket data."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path

from polymarket_playground.data.polymarket_client import PolymarketClient

logger = logging.getLogger(__name__)


class MarketCache:
    """Cache resolved Polymarket markets and their price histories in SQLite.

    Provides bulk sync from the Polymarket API and fast local reads for
    training episodes.
    """

    _CREATE_MARKETS = """
    CREATE TABLE IF NOT EXISTS markets (
        id          TEXT PRIMARY KEY,
        question    TEXT NOT NULL,
        outcome     REAL,
        volume      REAL NOT NULL DEFAULT 0,
        metadata    TEXT NOT NULL DEFAULT '{}',
        category    TEXT
    );
    """

    _CREATE_PRICES = """
    CREATE TABLE IF NOT EXISTS price_history (
        market_id   TEXT NOT NULL,
        timestamp   INTEGER NOT NULL,
        price       REAL NOT NULL,
        FOREIGN KEY (market_id) REFERENCES markets(id)
    );
    """

    _CREATE_PRICE_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_price_market
    ON price_history(market_id, timestamp);
    """

    def __init__(self, db_path: str | Path = "market_data.db") -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(self._CREATE_MARKETS)
        self._conn.execute(self._CREATE_PRICES)
        self._conn.execute(self._CREATE_PRICE_INDEX)
        self._migrate_category_column()
        self._conn.commit()
        self._client = PolymarketClient()

    def _migrate_category_column(self) -> None:
        """Add category column if it doesn't exist (migration for existing DBs)."""
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(markets)").fetchall()
        }
        if "category" not in cols:
            self._conn.execute("ALTER TABLE markets ADD COLUMN category TEXT")
            self._backfill_categories()

    def _backfill_categories(self) -> None:
        """Backfill categories for existing cached markets from stored metadata."""
        rows = self._conn.execute(
            "SELECT id, metadata FROM markets WHERE category IS NULL"
        ).fetchall()
        for row in rows:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            category = self._extract_category(meta)
            if category:
                self._conn.execute(
                    "UPDATE markets SET category = ? WHERE id = ?",
                    (category, row["id"]),
                )
        if rows:
            logger.info("Backfilled categories for %d markets", len(rows))

    @staticmethod
    def _extract_category(meta: dict) -> str | None:
        """Extract category from Polymarket metadata."""
        # Prefer groupItemTitle (event-level category)
        group = meta.get("groupItemTitle")
        if group:
            return group

        # Fall back to first tag
        tags = meta.get("tags")
        if tags:
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except (json.JSONDecodeError, TypeError):
                    tags = []
            if isinstance(tags, list) and tags:
                tag = tags[0]
                if isinstance(tag, dict):
                    return tag.get("label") or tag.get("slug")
                return str(tag)
        return None

    @staticmethod
    def _determine_outcome(market: dict) -> float | None:
        """Determine market outcome from resolution data.

        For resolved markets, outcomePrices will show 1.0/0.0 (or very close).
        """
        yes_price, no_price = PolymarketClient.parse_outcome_prices(market)

        # A resolved market will have prices at 1.0/0.0 or 0.0/1.0
        if yes_price >= 0.95:
            return 1.0
        if no_price >= 0.95:
            return 0.0

        # Check explicit resolution fields
        resolution = market.get("resolution")
        if resolution is not None:
            try:
                return float(resolution)
            except (ValueError, TypeError):
                pass

        # Not clearly resolved
        return None

    def sync(
        self,
        *,
        limit: int = 100,
        category: str | None = None,
        on_progress: callable | None = None,
    ) -> dict:
        """Fetch resolved markets from Polymarket and cache them locally.

        Skips markets already in the cache. Returns sync stats.
        """
        fetched = 0
        skipped = 0
        errors = 0
        offset = 0
        batch_size = 50

        while fetched + skipped < limit:
            remaining = limit - fetched - skipped
            batch_limit = min(batch_size, remaining + 10)  # fetch a bit extra for filtering

            try:
                markets = self._client.list_markets(
                    active=False,
                    closed=True,
                    limit=batch_limit,
                    offset=offset,
                    order="volume24hr",
                    ascending=False,
                    tag=category,
                )
            except Exception as e:
                logger.error("Failed to fetch markets at offset %d: %s", offset, e)
                break

            if not markets:
                break

            offset += len(markets)

            for m in markets:
                if fetched + skipped >= limit:
                    break

                market_id = str(m.get("id", ""))
                if not market_id:
                    continue

                # Skip if already cached
                if self._market_exists(market_id):
                    skipped += 1
                    continue

                # Need CLOB token IDs for price history
                yes_id, _ = PolymarketClient.parse_clob_token_ids(m)
                if not yes_id:
                    skipped += 1
                    continue

                # Fetch price history
                try:
                    history = self._client.get_price_history(
                        yes_id, interval="max", fidelity=60
                    )
                    if len(history) < 10:
                        history = self._client.get_price_history(
                            yes_id, interval="max", fidelity=5
                        )
                except Exception as e:
                    logger.debug("Price history failed for %s: %s", market_id, e)
                    errors += 1
                    continue

                if len(history) < 10:
                    skipped += 1
                    continue

                # Check for meaningful price movement
                prices = [float(h["p"]) for h in history]
                if max(prices) - min(prices) < 0.005:
                    skipped += 1
                    continue

                # Determine outcome from resolution data
                outcome = self._determine_outcome(m)

                # Extract category
                cat = self._extract_category(m)

                with self._lock:
                    # Store market
                    self._conn.execute(
                        "INSERT OR REPLACE INTO markets (id, question, outcome, volume, metadata, category) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            market_id,
                            m.get("question", ""),
                            outcome,
                            float(m.get("volume", 0) or 0),
                            json.dumps(m, default=str),
                            cat,
                        ),
                    )

                    # Store price history
                    self._conn.executemany(
                        "INSERT INTO price_history (market_id, timestamp, price) VALUES (?, ?, ?)",
                        [(market_id, int(h["t"]), float(h["p"])) for h in history],
                    )

                fetched += 1
                logger.debug("Cached %s (%d points)", market_id, len(history))

                if on_progress:
                    on_progress(fetched, skipped, errors)

            with self._lock:
                self._conn.commit()

        stats = {"fetched": fetched, "skipped": skipped, "errors": errors}
        logger.info("Sync complete: %s", stats)
        return stats

    def get_markets(self) -> list[dict]:
        """Return all cached markets."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, question, outcome, volume, category FROM markets ORDER BY volume DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_market_metadata(self, market_id: str) -> dict | None:
        """Return full metadata JSON for a market."""
        with self._lock:
            row = self._conn.execute(
                "SELECT metadata FROM markets WHERE id = ?", (market_id,)
            ).fetchone()
        if row:
            return json.loads(row["metadata"])
        return None

    def get_market_ids_by_category(self, category: str) -> list[str]:
        """Return market IDs matching the given category."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM markets WHERE category = ? ORDER BY volume DESC",
                (category,),
            ).fetchall()
        return [row["id"] for row in rows]

    def get_price_history(self, market_id: str) -> list[dict]:
        """Return price history for a market as list of {t, p} dicts."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT timestamp, price FROM price_history WHERE market_id = ? ORDER BY timestamp",
                (market_id,),
            ).fetchall()
        return [{"t": r["timestamp"], "p": r["price"]} for r in rows]

    def stats(self) -> dict:
        """Return cache statistics: total markets, total price points, categories."""
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) AS cnt FROM markets").fetchone()["cnt"]
            points = self._conn.execute("SELECT COUNT(*) AS cnt FROM price_history").fetchone()["cnt"]
            cat_rows = self._conn.execute(
                "SELECT COALESCE(category, 'uncategorized') AS cat, COUNT(*) AS cnt "
                "FROM markets GROUP BY cat ORDER BY cnt DESC"
            ).fetchall()
        categories = {row["cat"]: row["cnt"] for row in cat_rows}

        return {
            "total_markets": total,
            "total_price_points": points,
            "categories": categories,
        }

    def get_outcome(self, market_id: str) -> float | None:
        """Return the resolution outcome for a market, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT outcome FROM markets WHERE id = ?", (market_id,)
            ).fetchone()
        if row and row["outcome"] is not None:
            return float(row["outcome"])
        return None

    def get_market_ids_by_date_range(
        self,
        before: int | None = None,
        after: int | None = None,
    ) -> list[str]:
        """Return market IDs whose price history falls within a date range.

        Parameters
        ----------
        before : int, optional
            Max timestamp (exclusive). Markets with all prices before this.
        after : int, optional
            Min timestamp (inclusive). Markets with any price at or after this.

        Only returns markets with known outcomes (resolution != None).
        """
        query = (
            "SELECT DISTINCT m.id FROM markets m "
            "JOIN price_history p ON m.id = p.market_id "
            "WHERE m.outcome IS NOT NULL"
        )
        params: list = []

        if before is not None:
            query += " AND p.timestamp < ?"
            params.append(before)
        if after is not None:
            query += " AND p.timestamp >= ?"
            params.append(after)

        query += " ORDER BY m.id"

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [row[0] for row in rows]

    def get_earliest_latest_timestamps(self) -> tuple[int, int]:
        """Return (earliest, latest) price timestamps across all markets."""
        with self._lock:
            row = self._conn.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM price_history"
            ).fetchone()
        if row and row[0] is not None:
            return (row[0], row[1])
        return (0, 0)

    def _market_exists(self, market_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM markets WHERE id = ? LIMIT 1", (market_id,)
            ).fetchone()
        return row is not None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
