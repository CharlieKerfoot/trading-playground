"""SessionLog — persistent SQLite logging for live/paper trading sessions."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DB = Path("sessions.db")


@dataclass
class SessionRecord:
  """Summary of a completed session."""
  session_id: int
  agent_type: str
  mode: str
  started_at: float
  ended_at: float | None
  starting_capital: float
  final_capital: float | None
  total_pnl: float | None
  markets_traded: int
  total_ticks: int
  risk_interventions: int
  config: dict


class SessionLog:
  """SQLite-backed session logger."""

  def __init__(self, db_path: Path | str = DEFAULT_DB):
    self._db = sqlite3.connect(str(db_path))
    self._db.row_factory = sqlite3.Row
    self._init_tables()
    self._session_id: int | None = None

  def _init_tables(self):
    self._db.executescript("""
      CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        agent_type TEXT NOT NULL,
        mode TEXT NOT NULL,
        started_at REAL NOT NULL,
        ended_at REAL,
        starting_capital REAL NOT NULL,
        final_capital REAL,
        total_pnl REAL,
        markets_traded INTEGER DEFAULT 0,
        total_ticks INTEGER DEFAULT 0,
        risk_interventions INTEGER DEFAULT 0,
        config TEXT DEFAULT '{}'
      );
      CREATE TABLE IF NOT EXISTS ticks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        timestamp REAL NOT NULL,
        market_id TEXT NOT NULL,
        yes_price REAL,
        no_price REAL,
        spread REAL,
        action TEXT,
        action_size REAL,
        fill_direction TEXT,
        fill_price REAL,
        fill_size REAL,
        fill_fees REAL,
        position_yes_shares REAL,
        position_no_shares REAL,
        position_cost_basis REAL,
        portfolio_capital REAL,
        portfolio_exposure REAL,
        risk_blocked INTEGER DEFAULT 0,
        risk_reason TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
      );
      CREATE INDEX IF NOT EXISTS idx_ticks_session ON ticks(session_id);
      CREATE INDEX IF NOT EXISTS idx_ticks_market ON ticks(session_id, market_id);
    """)
    self._db.commit()

  def start_session(
    self,
    agent_type: str,
    mode: str,
    starting_capital: float,
    config: dict | None = None,
  ) -> int:
    cur = self._db.execute(
      """INSERT INTO sessions (agent_type, mode, started_at, starting_capital, config)
         VALUES (?, ?, ?, ?, ?)""",
      (agent_type, mode, time.time(), starting_capital, json.dumps(config or {})),
    )
    self._db.commit()
    self._session_id = cur.lastrowid
    logger.info("Session %d started: agent=%s mode=%s", self._session_id, agent_type, mode)
    return self._session_id

  def log_tick(
    self,
    market_id: str,
    state: dict,
    action: dict | None = None,
    fill: dict | None = None,
    position: dict | None = None,
    portfolio: dict | None = None,
    risk_blocked: bool = False,
    risk_reason: str | None = None,
  ) -> None:
    if self._session_id is None:
      return
    action = action or {}
    fill = fill or {}
    position = position or {}
    portfolio = portfolio or {}

    self._db.execute(
      """INSERT INTO ticks (
        session_id, timestamp, market_id,
        yes_price, no_price, spread,
        action, action_size,
        fill_direction, fill_price, fill_size, fill_fees,
        position_yes_shares, position_no_shares, position_cost_basis,
        portfolio_capital, portfolio_exposure,
        risk_blocked, risk_reason
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
      (
        self._session_id, time.time(), market_id,
        state.get("yes_price"), state.get("no_price"), state.get("spread"),
        action.get("direction"), action.get("size"),
        fill.get("direction"), fill.get("price"), fill.get("size"), fill.get("fees"),
        position.get("yes_shares", 0), position.get("no_shares", 0),
        position.get("cost_basis", 0),
        portfolio.get("available_capital"), portfolio.get("total_exposure"),
        int(risk_blocked), risk_reason,
      ),
    )
    self._db.commit()

  def end_session(
    self,
    final_capital: float,
    total_pnl: float,
    markets_traded: int,
    total_ticks: int,
    risk_interventions: int,
  ) -> None:
    if self._session_id is None:
      return
    self._db.execute(
      """UPDATE sessions SET
        ended_at = ?, final_capital = ?, total_pnl = ?,
        markets_traded = ?, total_ticks = ?, risk_interventions = ?
       WHERE id = ?""",
      (
        time.time(), final_capital, total_pnl,
        markets_traded, total_ticks, risk_interventions,
        self._session_id,
      ),
    )
    self._db.commit()
    logger.info(
      "Session %d ended: pnl=%+.4f ticks=%d",
      self._session_id, total_pnl, total_ticks,
    )

  def list_sessions(self, limit: int = 20) -> list[SessionRecord]:
    rows = self._db.execute(
      "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [
      SessionRecord(
        session_id=r["id"],
        agent_type=r["agent_type"],
        mode=r["mode"],
        started_at=r["started_at"],
        ended_at=r["ended_at"],
        starting_capital=r["starting_capital"],
        final_capital=r["final_capital"],
        total_pnl=r["total_pnl"],
        markets_traded=r["markets_traded"],
        total_ticks=r["total_ticks"],
        risk_interventions=r["risk_interventions"],
        config=json.loads(r["config"] or "{}"),
      )
      for r in rows
    ]

  def get_session_ticks(self, session_id: int) -> list[dict]:
    rows = self._db.execute(
      "SELECT * FROM ticks WHERE session_id = ? ORDER BY id", (session_id,)
    ).fetchall()
    return [dict(r) for r in rows]

  def close(self):
    self._db.close()
