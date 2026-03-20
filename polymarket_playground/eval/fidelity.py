"""Sim-to-Real Fidelity Validator — compare paper fills vs real orderbook."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path

from polymarket_playground.core.state import MarketState
from polymarket_playground.core.types import Fill

logger = logging.getLogger(__name__)


@dataclass
class OrderbookSnapshot:
    """A snapshot of the orderbook at a point in time."""

    market_id: str
    timestamp: float
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    book_depth: float
    spread: float


@dataclass
class FidelityRecord:
    """Comparison of a paper fill vs what the real book would have given."""

    market_id: str
    timestamp: float
    direction: str
    size: float
    paper_price: float
    book_bid: float
    book_ask: float
    slippage_error: float  # paper_price - expected_real_price


@dataclass
class FidelityReport:
    """Aggregate fidelity metrics."""

    n_records: int = 0
    mean_slippage_error: float = 0.0
    max_slippage_error: float = 0.0
    mean_abs_error: float = 0.0
    fidelity_score: float = 0.0  # 0-1, higher is better
    records: list[FidelityRecord] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Fidelity Score: {self.fidelity_score:.4f} "
            f"({self.n_records} fills compared)\n"
            f"  Mean slippage error: {self.mean_slippage_error:+.6f}\n"
            f"  Mean |error|:        {self.mean_abs_error:.6f}\n"
            f"  Max  |error|:        {self.max_slippage_error:.6f}"
        )


class FidelityValidator:
    """Records orderbook snapshots during paper trading and compares fills.

    Usage:
        1. Call ``record_snapshot(state)`` each tick during live paper runs
        2. Call ``record_fill(fill, state)`` when a paper fill occurs
        3. Call ``report()`` to get fidelity metrics
    """

    def __init__(self, db_path: str | Path = "market_data.db") -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        with self._lock:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    yes_bid REAL NOT NULL,
                    yes_ask REAL NOT NULL,
                    no_bid REAL NOT NULL,
                    no_ask REAL NOT NULL,
                    book_depth REAL NOT NULL,
                    spread REAL NOT NULL
                )"""
            )
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS paper_fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    direction TEXT NOT NULL,
                    size REAL NOT NULL,
                    paper_price REAL NOT NULL,
                    book_bid REAL NOT NULL,
                    book_ask REAL NOT NULL,
                    slippage_error REAL NOT NULL
                )"""
            )
            self._conn.commit()

    def record_snapshot(self, state: MarketState) -> None:
        """Record a live orderbook snapshot."""
        import time

        ts = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO orderbook_snapshots "
                "(market_id, timestamp, yes_bid, yes_ask, no_bid, no_ask, book_depth, spread) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    state.market_id, ts,
                    state.yes_bid, state.yes_ask,
                    state.no_bid, state.no_ask,
                    state.book_depth, state.spread,
                ),
            )
            self._conn.commit()

    def record_fill(self, fill: Fill, state: MarketState) -> FidelityRecord:
        """Record a paper fill and compute slippage error vs real book.

        The slippage error is the difference between the paper fill price
        and the price the real book would have given (ask for buys, bid for sells).
        """
        import time

        ts = time.time()

        if fill.direction == "yes":
            expected_price = state.yes_ask
        elif fill.direction == "no":
            expected_price = state.no_ask
        else:  # close
            expected_price = state.yes_bid

        slippage_error = fill.price - expected_price

        record = FidelityRecord(
            market_id=state.market_id,
            timestamp=ts,
            direction=fill.direction,
            size=fill.size,
            paper_price=fill.price,
            book_bid=state.yes_bid if fill.direction != "no" else state.no_bid,
            book_ask=state.yes_ask if fill.direction != "no" else state.no_ask,
            slippage_error=slippage_error,
        )

        with self._lock:
            self._conn.execute(
                "INSERT INTO paper_fills "
                "(market_id, timestamp, direction, size, paper_price, "
                "book_bid, book_ask, slippage_error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.market_id, record.timestamp,
                    record.direction, record.size,
                    record.paper_price, record.book_bid,
                    record.book_ask, record.slippage_error,
                ),
            )
            self._conn.commit()

        return record

    def report(self) -> FidelityReport:
        """Generate a fidelity report from all recorded fills."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM paper_fills ORDER BY timestamp"
            ).fetchall()

        if not rows:
            return FidelityReport()

        records = []
        errors = []
        for row in rows:
            rec = FidelityRecord(
                market_id=row["market_id"],
                timestamp=row["timestamp"],
                direction=row["direction"],
                size=row["size"],
                paper_price=row["paper_price"],
                book_bid=row["book_bid"],
                book_ask=row["book_ask"],
                slippage_error=row["slippage_error"],
            )
            records.append(rec)
            errors.append(rec.slippage_error)

        abs_errors = [abs(e) for e in errors]
        mean_error = sum(errors) / len(errors)
        mean_abs = sum(abs_errors) / len(abs_errors)
        max_abs = max(abs_errors)

        # Fidelity score: 1.0 = perfect, decays with error magnitude
        # Score = 1 / (1 + 10 * mean_abs_error)
        fidelity_score = 1.0 / (1.0 + 10.0 * mean_abs)

        return FidelityReport(
            n_records=len(records),
            mean_slippage_error=mean_error,
            max_slippage_error=max_abs,
            mean_abs_error=mean_abs,
            fidelity_score=fidelity_score,
            records=records,
        )

    def clear(self) -> None:
        """Clear all recorded data."""
        with self._lock:
            self._conn.execute("DELETE FROM orderbook_snapshots")
            self._conn.execute("DELETE FROM paper_fills")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
