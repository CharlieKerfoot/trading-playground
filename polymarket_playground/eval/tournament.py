"""Agent Tournament with ELO ranking system."""

from __future__ import annotations

import json
import logging
import math
import random
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path

from polymarket_playground.core.types import EpisodeResult

logger = logging.getLogger(__name__)

# Default ELO starting rating
DEFAULT_ELO = 1500
K_FACTOR = 32


@dataclass
class TournamentResult:
    """Result of a single head-to-head match."""

    agent_a: str
    agent_b: str
    pnl_a: float
    pnl_b: float
    winner: str  # agent name or "draw"
    market_id: str = ""


@dataclass
class RankingEntry:
    """An agent's ELO ranking."""

    agent_name: str
    elo: float
    matches: int
    wins: int
    losses: int
    draws: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.matches if self.matches > 0 else 0.0


class Tournament:
    """Runs head-to-head matches between agents and maintains ELO rankings.

    Rankings are persisted to SQLite so they survive across sessions.
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
                """CREATE TABLE IF NOT EXISTS elo_rankings (
                    agent_name TEXT PRIMARY KEY,
                    elo REAL NOT NULL DEFAULT 1500,
                    matches INTEGER NOT NULL DEFAULT 0,
                    wins INTEGER NOT NULL DEFAULT 0,
                    losses INTEGER NOT NULL DEFAULT 0,
                    draws INTEGER NOT NULL DEFAULT 0
                )"""
            )
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS match_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    agent_a TEXT NOT NULL,
                    agent_b TEXT NOT NULL,
                    pnl_a REAL NOT NULL,
                    pnl_b REAL NOT NULL,
                    winner TEXT NOT NULL,
                    market_id TEXT
                )"""
            )
            self._conn.commit()

    def run_match(
        self,
        env,
        agent_a_name: str,
        agent_a,
        agent_b_name: str,
        agent_b,
        n_episodes: int = 10,
    ) -> list[TournamentResult]:
        """Run a head-to-head match between two agents on identical seeded episodes.

        Returns a list of per-episode results.
        """
        from polymarket_playground.eval.runner import EpisodeRunner

        seeds = [random.randint(0, 2**31 - 1) for _ in range(n_episodes)]
        results = []

        runner_a = EpisodeRunner(env, agent_a)
        runner_b = EpisodeRunner(env, agent_b)

        results_a = runner_a.run_seeded(seeds)
        results_b = runner_b.run_seeded(seeds)

        for ep_a, ep_b in zip(results_a, results_b):
            pnl_a = ep_a.total_reward
            pnl_b = ep_b.total_reward

            if abs(pnl_a - pnl_b) < 1e-6:
                winner = "draw"
            elif pnl_a > pnl_b:
                winner = agent_a_name
            else:
                winner = agent_b_name

            result = TournamentResult(
                agent_a=agent_a_name,
                agent_b=agent_b_name,
                pnl_a=pnl_a,
                pnl_b=pnl_b,
                winner=winner,
            )
            results.append(result)

        # Update ELO ratings based on aggregate result
        wins_a = sum(1 for r in results if r.winner == agent_a_name)
        wins_b = sum(1 for r in results if r.winner == agent_b_name)
        draws = sum(1 for r in results if r.winner == "draw")

        # Score: 1 for each win, 0.5 for each draw
        score_a = wins_a + 0.5 * draws
        score_b = wins_b + 0.5 * draws
        total = len(results)

        self._update_elo(agent_a_name, agent_b_name, score_a / total, score_b / total)
        self._record_results(agent_a_name, agent_b_name, results, wins_a, wins_b, draws)

        return results

    def _update_elo(
        self,
        agent_a: str,
        agent_b: str,
        score_a: float,
        score_b: float,
    ) -> None:
        """Update ELO ratings for both agents after a match."""
        elo_a = self._get_elo(agent_a)
        elo_b = self._get_elo(agent_b)

        expected_a = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
        expected_b = 1.0 - expected_a

        new_elo_a = elo_a + K_FACTOR * (score_a - expected_a)
        new_elo_b = elo_b + K_FACTOR * (score_b - expected_b)

        self._set_elo(agent_a, new_elo_a)
        self._set_elo(agent_b, new_elo_b)

    def _get_elo(self, agent_name: str) -> float:
        with self._lock:
            row = self._conn.execute(
                "SELECT elo FROM elo_rankings WHERE agent_name = ?",
                (agent_name,),
            ).fetchone()
        if row:
            return float(row["elo"])
        return DEFAULT_ELO

    def _set_elo(self, agent_name: str, elo: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO elo_rankings (agent_name, elo) VALUES (?, ?) "
                "ON CONFLICT(agent_name) DO UPDATE SET elo = ?",
                (agent_name, elo, elo),
            )
            self._conn.commit()

    def _record_results(
        self,
        agent_a: str,
        agent_b: str,
        results: list[TournamentResult],
        wins_a: int,
        wins_b: int,
        draws: int,
    ) -> None:
        import time

        ts = time.time()
        with self._lock:
            for r in results:
                self._conn.execute(
                    "INSERT INTO match_history "
                    "(timestamp, agent_a, agent_b, pnl_a, pnl_b, winner, market_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ts, r.agent_a, r.agent_b, r.pnl_a, r.pnl_b, r.winner, r.market_id),
                )

            # Update match stats for agent_a
            self._conn.execute(
                "INSERT INTO elo_rankings (agent_name, elo, matches, wins, losses, draws) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(agent_name) DO UPDATE SET "
                "matches = matches + ?, wins = wins + ?, losses = losses + ?, draws = draws + ?",
                (agent_a, DEFAULT_ELO, len(results), wins_a, wins_b, draws,
                 len(results), wins_a, wins_b, draws),
            )

            # Update match stats for agent_b
            self._conn.execute(
                "INSERT INTO elo_rankings (agent_name, elo, matches, wins, losses, draws) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(agent_name) DO UPDATE SET "
                "matches = matches + ?, wins = wins + ?, losses = losses + ?, draws = draws + ?",
                (agent_b, DEFAULT_ELO, len(results), wins_b, wins_a, draws,
                 len(results), wins_b, wins_a, draws),
            )

            self._conn.commit()

    def get_rankings(self) -> list[RankingEntry]:
        """Get all agent rankings sorted by ELO (descending)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM elo_rankings ORDER BY elo DESC"
            ).fetchall()
        return [
            RankingEntry(
                agent_name=row["agent_name"],
                elo=row["elo"],
                matches=row["matches"],
                wins=row["wins"],
                losses=row["losses"],
                draws=row["draws"],
            )
            for row in rows
        ]

    def reset(self) -> None:
        """Clear all rankings and match history."""
        with self._lock:
            self._conn.execute("DELETE FROM elo_rankings")
            self._conn.execute("DELETE FROM match_history")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
