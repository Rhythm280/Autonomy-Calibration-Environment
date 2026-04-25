"""
database.py — SQLite persistence layer for Autonomy Calibration Environment.

Uses stdlib sqlite3 only — no external dependencies.

Tables:
  episodes  — one row per episode (id, task, seed, start_time, end_time, total_reward)
  steps     — one row per environment step (episode_id, step_index, decision, reward, done)

Public API:
  init_db()                  — create tables (idempotent)
  create_episode(task, seed) — insert episode row, return episode_id
  log_step(...)              — insert step row
  close_episode(id, score)   — update episode with final score + end_time
  get_episode(id)            — fetch episode + all steps
  list_episodes(limit)       — list recent episodes
  replay_episode(id)         — return ordered step list for replay
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("AUTONOMY_ENV_DB", "autonomy_env.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    task          TEXT    NOT NULL,
    seed          INTEGER,
    started_at    TEXT    NOT NULL,
    ended_at      TEXT,
    total_reward  REAL    DEFAULT 0.0,
    done          INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS steps (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id    INTEGER NOT NULL REFERENCES episodes(id),
    step_index    INTEGER NOT NULL,
    decision      TEXT    NOT NULL,
    reward        REAL    NOT NULL,
    done          INTEGER NOT NULL DEFAULT 0,
    timestamp     TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_steps_episode ON steps(episode_id);
"""


# ─── Connection ───────────────────────────────────────────────────────────────

@contextmanager
def _conn(path: str = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """Context-managed SQLite connection with WAL mode for concurrent safety."""
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─── Init ─────────────────────────────────────────────────────────────────────

def init_db(path: str = DB_PATH) -> None:
    """Create all tables if they don't exist. Safe to call multiple times."""
    with _conn(path) as c:
        c.executescript(_SCHEMA)
    logger.info("DB: Initialised SQLite at %s", path)


# ─── Write ────────────────────────────────────────────────────────────────────

def create_episode(task: str, seed: int | None, path: str = DB_PATH) -> int:
    """Insert a new episode row. Returns the new episode_id."""
    _ensure(path)
    now = _now()
    with _conn(path) as c:
        cur = c.execute(
            "INSERT INTO episodes (task, seed, started_at) VALUES (?, ?, ?)",
            (task, seed, now),
        )
        eid = cur.lastrowid
    logger.debug("DB: Episode created id=%d task=%s seed=%s", eid, task, seed)
    return eid


def log_step(
    episode_id: int,
    step_index: int,
    decision: str,
    reward: float,
    done: bool,
    path: str = DB_PATH,
) -> None:
    """Record a single environment step."""
    with _conn(path) as c:
        c.execute(
            "INSERT INTO steps (episode_id, step_index, decision, reward, done, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (episode_id, step_index, decision, round(reward, 4), int(done), _now()),
        )


def close_episode(episode_id: int, total_reward: float, path: str = DB_PATH) -> None:
    """Mark episode as done and record final score."""
    with _conn(path) as c:
        c.execute(
            "UPDATE episodes SET ended_at=?, total_reward=?, done=1 WHERE id=?",
            (_now(), round(total_reward, 4), episode_id),
        )
    logger.debug("DB: Episode closed id=%d score=%.4f", episode_id, total_reward)


# ─── Read ─────────────────────────────────────────────────────────────────────

def list_episodes(limit: int = 20, path: str = DB_PATH) -> list[dict[str, Any]]:
    """Return the most recent `limit` episodes."""
    _ensure(path)
    with _conn(path) as c:
        rows = c.execute(
            "SELECT * FROM episodes ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_episode(episode_id: int, path: str = DB_PATH) -> dict[str, Any]:
    """Return full episode dict including all steps."""
    _ensure(path)
    with _conn(path) as c:
        ep = c.execute("SELECT * FROM episodes WHERE id=?", (episode_id,)).fetchone()
        if ep is None:
            raise ValueError(f"Episode {episode_id} not found.")
        steps = c.execute(
            "SELECT * FROM steps WHERE episode_id=? ORDER BY step_index ASC",
            (episode_id,),
        ).fetchall()
    return {
        "episode": dict(ep),
        "steps": [dict(s) for s in steps],
    }


def replay_episode(episode_id: int, path: str = DB_PATH) -> list[dict[str, Any]]:
    """Return ordered step list for replay — same as get_episode but steps only."""
    return get_episode(episode_id, path)["steps"]


# ─── Helpers ──────────────────────────────────────────────────────────────────

_initialised: set[str] = set()

def _ensure(path: str = DB_PATH) -> None:
    """Lazy init — create schema on first use."""
    if path not in _initialised:
        init_db(path)
        _initialised.add(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Auto-init on import
_ensure(DB_PATH)
