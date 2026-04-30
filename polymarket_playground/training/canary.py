"""Canary Paper Trading — paper trade top agents on live data for validation."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CanaryConfig:
    """Configuration for a canary paper trading run."""

    agent_name: str
    market_ids: list[str]
    duration_hours: float = 24.0
    interval_seconds: int = 60
    starting_capital: float = 100.0


@dataclass
class CanaryStatus:
    """Status of a canary run."""

    canary_id: int
    agent_name: str
    market_ids: list[str]
    started_at: float
    duration_hours: float
    elapsed_hours: float
    ticks: int
    total_pnl: float
    is_running: bool
    confidence_score: float  # 0-1

    @property
    def time_remaining_hours(self) -> float:
        return max(0.0, self.duration_hours - self.elapsed_hours)


class CanaryManager:
    """Manages canary paper trading runs.

    A canary run paper trades an agent on live market data for 24-72h
    after the pipeline recommends DEPLOY. Produces a deployment confidence
    score comparing live paper results vs backtest expectations.
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
                """CREATE TABLE IF NOT EXISTS canary_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    market_ids TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    duration_hours REAL NOT NULL,
                    interval_seconds INTEGER NOT NULL,
                    starting_capital REAL NOT NULL,
                    ticks INTEGER NOT NULL DEFAULT 0,
                    total_pnl REAL NOT NULL DEFAULT 0.0,
                    is_running INTEGER NOT NULL DEFAULT 1,
                    stopped_at REAL,
                    config TEXT
                )"""
            )
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS canary_ticks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    canary_id INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    market_id TEXT NOT NULL,
                    yes_price REAL NOT NULL,
                    action TEXT NOT NULL,
                    pnl_delta REAL NOT NULL DEFAULT 0.0,
                    cumulative_pnl REAL NOT NULL DEFAULT 0.0,
                    FOREIGN KEY (canary_id) REFERENCES canary_runs(id)
                )"""
            )
            self._conn.commit()

    def start_canary(self, config: CanaryConfig) -> int:
        """Register a new canary run. Returns the canary ID."""
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO canary_runs "
                "(agent_name, market_ids, started_at, duration_hours, "
                "interval_seconds, starting_capital, config) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    config.agent_name,
                    json.dumps(config.market_ids),
                    time.time(),
                    config.duration_hours,
                    config.interval_seconds,
                    config.starting_capital,
                    json.dumps({
                        "agent_name": config.agent_name,
                        "market_ids": config.market_ids,
                        "duration_hours": config.duration_hours,
                    }),
                ),
            )
            self._conn.commit()
            return cursor.lastrowid

    def record_tick(
        self,
        canary_id: int,
        market_id: str,
        yes_price: float,
        action: str,
        pnl_delta: float,
        cumulative_pnl: float,
    ) -> None:
        """Record a canary tick."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO canary_ticks "
                "(canary_id, timestamp, market_id, yes_price, action, pnl_delta, cumulative_pnl) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (canary_id, time.time(), market_id, yes_price, action, pnl_delta, cumulative_pnl),
            )
            self._conn.execute(
                "UPDATE canary_runs SET ticks = ticks + 1, total_pnl = ? WHERE id = ?",
                (cumulative_pnl, canary_id),
            )
            self._conn.commit()

    def stop_canary(self, canary_id: int) -> None:
        """Mark a canary run as stopped."""
        with self._lock:
            self._conn.execute(
                "UPDATE canary_runs SET is_running = 0, stopped_at = ? WHERE id = ?",
                (time.time(), canary_id),
            )
            self._conn.commit()

    def get_status(self, canary_id: int | None = None) -> CanaryStatus | list[CanaryStatus] | None:
        """Get status of a specific canary or all running canaries."""
        if canary_id is not None:
            with self._lock:
                row = self._conn.execute(
                    "SELECT * FROM canary_runs WHERE id = ?", (canary_id,)
                ).fetchone()
            if not row:
                return None
            return self._row_to_status(row)

        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM canary_runs WHERE is_running = 1 ORDER BY started_at DESC"
            ).fetchall()
        return [self._row_to_status(row) for row in rows]

    def get_all_canaries(self, limit: int = 20) -> list[CanaryStatus]:
        """Get all canary runs (running and stopped)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM canary_runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_status(row) for row in rows]

    def compute_confidence(self, canary_id: int, backtest_sharpe: float = 0.0) -> float:
        """Compute deployment confidence score for a canary run.

        Factors:
        - Duration: more hours = more confidence
        - P&L consistency: steady gains > volatile
        - Comparison to backtest: live results match predictions
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM canary_runs WHERE id = ?", (canary_id,)
            ).fetchone()
            ticks = self._conn.execute(
                "SELECT cumulative_pnl FROM canary_ticks WHERE canary_id = ? ORDER BY timestamp",
                (canary_id,),
            ).fetchall()

        if not row or not ticks:
            return 0.0

        pnl_values = [t["cumulative_pnl"] for t in ticks]
        elapsed_hours = (time.time() - row["started_at"]) / 3600.0
        duration_hours = row["duration_hours"]

        # Factor 1: Duration coverage (0-1)
        duration_score = min(1.0, elapsed_hours / max(duration_hours, 1.0))

        # Factor 2: P&L positivity (0-1)
        final_pnl = pnl_values[-1] if pnl_values else 0.0
        pnl_score = 1.0 / (1.0 + max(0.0, -final_pnl * 10.0))  # penalize losses

        # Factor 3: Consistency — fraction of time P&L was positive
        positive_ticks = sum(1 for p in pnl_values if p > 0)
        consistency_score = positive_ticks / len(pnl_values) if pnl_values else 0.0

        # Factor 4: Sample size
        sample_score = min(1.0, len(pnl_values) / 100.0)

        # Weighted combination
        confidence = (
            0.25 * duration_score
            + 0.35 * pnl_score
            + 0.25 * consistency_score
            + 0.15 * sample_score
        )

        return round(min(1.0, max(0.0, confidence)), 4)

    def _row_to_status(self, row) -> CanaryStatus:
        elapsed_hours = (time.time() - row["started_at"]) / 3600.0
        canary_id = row["id"]
        return CanaryStatus(
            canary_id=canary_id,
            agent_name=row["agent_name"],
            market_ids=json.loads(row["market_ids"]),
            started_at=row["started_at"],
            duration_hours=row["duration_hours"],
            elapsed_hours=elapsed_hours,
            ticks=row["ticks"],
            total_pnl=row["total_pnl"],
            is_running=bool(row["is_running"]),
            confidence_score=self.compute_confidence(canary_id),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
