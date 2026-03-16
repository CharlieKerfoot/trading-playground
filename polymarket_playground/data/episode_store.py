"""EpisodeStore — SQLite-backed persistent episode cache."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from polymarket_playground.core.types import Episode, EpisodeResult


class EpisodeStore:
    """Persist episodes and their results to a local SQLite database.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Created if it does not exist.
    """

    _CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS episodes (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        market_id   TEXT    NOT NULL,
        market_type TEXT    NOT NULL,
        episode     TEXT    NOT NULL,
        results     TEXT    NOT NULL,
        run_id      TEXT,
        created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
    );
    """

    _CREATE_RUNS_TABLE = """
    CREATE TABLE IF NOT EXISTS training_runs (
        id              TEXT PRIMARY KEY,
        agent_type      TEXT NOT NULL,
        agent_config    TEXT NOT NULL DEFAULT '{}',
        market_filter   TEXT,
        num_episodes    INTEGER NOT NULL,
        status          TEXT NOT NULL DEFAULT 'pending',
        created_at      TEXT NOT NULL DEFAULT (datetime('now')),
        started_at      TEXT,
        finished_at     TEXT,
        error           TEXT
    );
    """

    def __init__(self, db_path: str | Path = "episodes.db") -> None:
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(self._CREATE_TABLE)
        self._conn.execute(self._CREATE_RUNS_TABLE)
        # Add run_id column if upgrading from old schema
        try:
            self._conn.execute("ALTER TABLE episodes ADD COLUMN run_id TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def save_episode(
        self, episode: Episode, results: EpisodeResult, run_id: str | None = None
    ) -> None:
        """Serialise and store an episode with its results."""
        episode_json = json.dumps(asdict(episode), default=str)
        results_json = json.dumps(asdict(results), default=str)
        self._conn.execute(
            "INSERT INTO episodes (market_id, market_type, episode, results, run_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (episode.market_id, episode.market_type, episode_json, results_json, run_id),
        )
        self._conn.commit()

    def save_run(self, run_data: dict) -> None:
        """Persist a training run record."""
        self._conn.execute(
            "INSERT OR REPLACE INTO training_runs "
            "(id, agent_type, agent_config, market_filter, num_episodes, status, "
            "created_at, started_at, finished_at, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_data["id"],
                run_data["agent_type"],
                json.dumps(run_data.get("agent_config", {})),
                run_data.get("market_filter"),
                run_data["num_episodes"],
                run_data["status"],
                run_data.get("created_at"),
                run_data.get("started_at"),
                run_data.get("finished_at"),
                run_data.get("error"),
            ),
        )
        self._conn.commit()

    def load_runs(self) -> list[dict]:
        """Load all training run records."""
        rows = self._conn.execute(
            "SELECT * FROM training_runs ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def load_episodes(
        self, filters: dict | None = None
    ) -> list[dict]:
        """Load stored episodes, optionally filtered by column values.

        Parameters
        ----------
        filters:
            Mapping of column name to required value.  Only ``market_id``
            and ``market_type`` are supported as filter keys.

        Returns
        -------
        list[dict]
            Each dict contains ``episode`` and ``results`` as parsed
            Python objects, plus the ``id`` and ``created_at`` metadata.
        """
        query = "SELECT * FROM episodes"
        params: list = []

        if filters:
            clauses = []
            for key in ("market_id", "market_type"):
                if key in filters:
                    clauses.append(f"{key} = ?")
                    params.append(filters[key])
            if clauses:
                query += " WHERE " + " AND ".join(clauses)

        rows = self._conn.execute(query, params).fetchall()
        results: list[dict] = []
        for row in rows:
            results.append(
                {
                    "id": row["id"],
                    "created_at": row["created_at"],
                    "episode": json.loads(row["episode"]),
                    "results": json.loads(row["results"]),
                }
            )
        return results

    def count(self) -> int:
        """Return the total number of stored episodes."""
        row = self._conn.execute("SELECT COUNT(*) AS cnt FROM episodes").fetchone()
        return row["cnt"]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
