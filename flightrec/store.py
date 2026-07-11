"""
SQLite trace store — schema, CRUD, and query helpers for Flight Recorder.
Local-first: everything stored in a single SQLite file (default: ./flightrec.db).
"""

from __future__ import annotations

import sqlite3
import json
import time
from typing import Optional, Any
from dataclasses import asdict

from flightrec.sdk import RunRecord, LLMCallRecord, ToolCallRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    agent_name TEXT DEFAULT '',
    status TEXT DEFAULT 'running',
    tags TEXT DEFAULT '',
    metadata_json TEXT DEFAULT '{}',
    total_tokens_in INTEGER DEFAULT 0,
    total_tokens_out INTEGER DEFAULT 0,
    total_cost REAL DEFAULT 0.0,
    total_latency_ms REAL DEFAULT 0.0,
    step_count INTEGER DEFAULT 0,
    started_at REAL NOT NULL,
    ended_at REAL
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    step_index INTEGER DEFAULT 0,
    model TEXT NOT NULL,
    provider TEXT DEFAULT 'anthropic',
    prompt TEXT DEFAULT '',
    response TEXT DEFAULT '',
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    cost REAL DEFAULT 0.0,
    latency_ms REAL DEFAULT 0.0,
    timestamp REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    step_index INTEGER DEFAULT 0,
    name TEXT NOT NULL,
    input_json TEXT DEFAULT '{}',
    output_json TEXT DEFAULT '{}',
    success INTEGER DEFAULT 1,
    error TEXT DEFAULT '',
    latency_ms REAL DEFAULT 0.0,
    timestamp REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS golden_tasks (
    id TEXT PRIMARY KEY,
    suite_name TEXT NOT NULL,
    task_name TEXT NOT NULL,
    description TEXT DEFAULT '',
    input_json TEXT DEFAULT '{}',
    expected_output TEXT DEFAULT '',
    checks_json TEXT DEFAULT '[]',
    judge_prompt TEXT DEFAULT '',
    enabled INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS eval_results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    task_id TEXT NOT NULL,
    passed INTEGER DEFAULT 0,
    score REAL DEFAULT 0.0,
    details_json TEXT DEFAULT '{}',
    timestamp REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_run ON tool_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_name ON runs(name);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);
CREATE INDEX IF NOT EXISTS idx_eval_results_run ON eval_results(run_id);
"""


class TraceStore:
    """SQLite-backed store for Flight Recorder traces."""

    def __init__(self, db_path: str = "flightrec.db"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row  # enables dict(row) in all queries
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)
            self._conn.commit()
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Save ──

    def save_run(
        self,
        run: RunRecord,
        llm_calls: list[LLMCallRecord],
        tool_calls: list[ToolCallRecord],
    ) -> str:
        """Persist a full run with all its LLM and tool calls."""
        c = self.conn
        c.execute(
            """INSERT OR REPLACE INTO runs
               (id, name, agent_name, status, tags, metadata_json,
                total_tokens_in, total_tokens_out, total_cost, total_latency_ms,
                step_count, started_at, ended_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run.id, run.name, run.agent_name, run.status, run.tags, run.metadata_json,
                run.total_tokens_in, run.total_tokens_out, run.total_cost, run.total_latency_ms,
                run.step_count, run.started_at, run.ended_at,
            ),
        )
        for llm in llm_calls:
            self.save_llm_call(llm)
        for tool in tool_calls:
            self.save_tool_call(tool)
        c.commit()
        return run.id

    def save_llm_call(self, call: LLMCallRecord) -> None:
        c = self.conn
        c.execute(
            """INSERT OR REPLACE INTO llm_calls
               (id, run_id, step_index, model, provider, prompt, response,
                tokens_in, tokens_out, cost, latency_ms, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                call.id, call.run_id, call.step_index, call.model, call.provider,
                call.prompt, call.response,
                call.tokens_in, call.tokens_out, call.cost, call.latency_ms, call.timestamp,
            ),
        )

    def save_tool_call(self, call: ToolCallRecord) -> None:
        c = self.conn
        c.execute(
            """INSERT OR REPLACE INTO tool_calls
               (id, run_id, step_index, name, input_json, output_json,
                success, error, latency_ms, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                call.id, call.run_id, call.step_index, call.name,
                call.input_json, call.output_json,
                1 if call.success else 0, call.error, call.latency_ms, call.timestamp,
            ),
        )

    # ── Query ──

    def get_run(self, run_id: str) -> Optional[dict]:
        """Get a run by ID with all its calls."""
        c = self.conn
        row = c.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        run = dict(row)
        run["llm_calls"] = [
            dict(r)
            for r in c.execute(
                "SELECT * FROM llm_calls WHERE run_id = ? ORDER BY step_index",
                (run_id,),
            ).fetchall()
        ]
        run["tool_calls"] = [
            dict(r)
            for r in c.execute(
                "SELECT * FROM tool_calls WHERE run_id = ? ORDER BY step_index",
                (run_id,),
            ).fetchall()
        ]
        return run

    def list_runs(
        self,
        limit: int = 20,
        offset: int = 0,
        agent_name: str = "",
        status: str = "",
    ) -> list[dict]:
        """List recent runs, optionally filtered."""
        c = self.conn
        query = "SELECT * FROM runs WHERE 1=1"
        params: list[Any] = []
        if agent_name:
            query += " AND agent_name = ?"
            params.append(agent_name)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        return [dict(r) for r in c.execute(query, params).fetchall()]

    def search_runs(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search across run names."""
        c = self.conn
        rows = c.execute(
            "SELECT * FROM runs WHERE name LIKE ? ORDER BY started_at DESC LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        """Aggregate statistics across all runs."""
        c = self.conn
        total = c.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        completed = c.execute(
            "SELECT COUNT(*) FROM runs WHERE status = 'completed'"
        ).fetchone()[0]
        failed = c.execute(
            "SELECT COUNT(*) FROM runs WHERE status = 'failed'"
        ).fetchone()[0]
        cost = c.execute("SELECT COALESCE(SUM(total_cost), 0) FROM runs").fetchone()[0]
        tokens = c.execute(
            "SELECT COALESCE(SUM(total_tokens_in + total_tokens_out), 0) FROM runs"
        ).fetchone()[0]
        latency = c.execute(
            "SELECT COALESCE(AVG(total_latency_ms), 0) FROM runs WHERE status = 'completed'"
        ).fetchone()[0]

        return {
            "total_runs": total,
            "completed_runs": completed,
            "failed_runs": failed,
            "total_cost": cost,
            "total_tokens": tokens,
            "avg_latency_ms": latency,
        }

    def delete_run(self, run_id: str) -> bool:
        """Delete a run and its calls (cascading)."""
        c = self.conn
        cur = c.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        c.commit()
        return cur.rowcount > 0

    # ── Golden tasks ──

    def save_golden_task(self, task: dict) -> str:
        """Save or update a golden task definition."""
        c = self.conn
        task_id = task.get("id", f"task-{int(time.time())}")
        c.execute(
            """INSERT OR REPLACE INTO golden_tasks
               (id, suite_name, task_name, description, input_json,
                expected_output, checks_json, judge_prompt, enabled)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                task.get("suite_name", "default"),
                task.get("task_name", ""),
                task.get("description", ""),
                json.dumps(task.get("input", {})),
                task.get("expected_output", ""),
                json.dumps(task.get("checks", [])),
                task.get("judge_prompt", ""),
                1 if task.get("enabled", True) else 0,
            ),
        )
        c.commit()
        return task_id

    def get_golden_tasks(self, suite_name: str = "") -> list[dict]:
        """Get all golden tasks, optionally filtered by suite."""
        c = self.conn
        if suite_name:
            rows = c.execute(
                "SELECT * FROM golden_tasks WHERE suite_name = ? AND enabled = 1",
                (suite_name,),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM golden_tasks WHERE enabled = 1"
            ).fetchall()
        return [dict(r) for r in rows]

    def save_eval_result(self, result: dict) -> None:
        """Save an evaluation result for a run/task pair."""
        c = self.conn
        c.execute(
            """INSERT OR REPLACE INTO eval_results
               (id, run_id, task_id, passed, score, details_json, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                result.get("id", f"eval-{int(time.time())}"),
                result["run_id"],
                result["task_id"],
                1 if result.get("passed", False) else 0,
                result.get("score", 0.0),
                json.dumps(result.get("details", {})),
                result.get("timestamp", time.time()),
            ),
        )
        c.commit()

    def get_eval_results(self, run_id: str) -> list[dict]:
        """Get all evaluation results for a run."""
        c = self.conn
        return [
            dict(r)
            for r in c.execute(
                "SELECT * FROM eval_results WHERE run_id = ?", (run_id,)
            ).fetchall()
        ]
