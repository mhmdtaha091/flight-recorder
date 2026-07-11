"""
FastAPI server — REST API for the Flight Recorder dashboard.
v3 per AGENT_FLIGHT_RECORDER_PLAN.md.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from flightrec.store import TraceStore


def create_app(store: TraceStore) -> FastAPI:
    app = FastAPI(
        title="Flight Recorder API",
        version="0.1.0",
        description="Local-first evaluation, tracing & replay harness for AI agents",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Runs ──

    @app.get("/api/runs")
    async def list_runs(
        limit: int = Query(20, ge=1, le=200),
        offset: int = Query(0, ge=0),
        agent: str = Query(""),
        status: str = Query(""),
    ):
        runs = store.list_runs(limit=limit, offset=offset, agent_name=agent, status=status)
        return {"runs": runs, "total": len(runs)}

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str):
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.delete("/api/runs/{run_id}")
    async def delete_run(run_id: str):
        if store.delete_run(run_id):
            return {"deleted": run_id}
        raise HTTPException(status_code=404, detail="Run not found")

    # ── Stats ──

    @app.get("/api/stats")
    async def get_stats():
        return store.get_stats()

    @app.get("/api/stats/trends")
    async def get_trends(days: int = Query(7, ge=1, le=90)):
        """Return daily cost and run-count trends."""
        conn = store.conn
        cutoff = time.time() - days * 86400
        rows = conn.execute(
            """SELECT date(started_at, 'unixepoch') as day,
                      COUNT(*) as runs,
                      SUM(total_cost) as cost,
                      AVG(total_latency_ms) as avg_latency
               FROM runs
               WHERE started_at >= ?
               GROUP BY day
               ORDER BY day""",
            (cutoff,),
        ).fetchall()
        return {
            "trends": [
                {
                    "day": r[0],
                    "runs": r[1],
                    "cost": round(r[2], 4) if r[2] else 0,
                    "avg_latency_ms": round(r[3], 0) if r[3] else 0,
                }
                for r in rows
            ]
        }

    # ── Search ──

    @app.get("/api/search")
    async def search_runs(q: str = Query("", min_length=1), limit: int = Query(20)):
        runs = store.search_runs(q, limit=limit)
        return {"runs": runs, "query": q}

    # ── Golden Tasks ──

    @app.get("/api/tasks")
    async def list_tasks(suite: str = Query("")):
        tasks = store.get_golden_tasks(suite)
        return {"tasks": tasks}

    @app.post("/api/tasks")
    async def create_task(task: dict):
        task_id = store.save_golden_task(task)
        return {"id": task_id}

    # ── Eval Results ──

    @app.get("/api/runs/{run_id}/evals")
    async def get_eval_results(run_id: str):
        results = store.get_eval_results(run_id)
        return {"evaluations": results}

    # ── Health ──

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "db": store.db_path}

    # ── Serve dashboard static files ──

    import os
    dashboard_dir = os.path.join(os.path.dirname(__file__), "..", "dashboard")
    if os.path.isdir(dashboard_dir):
        app.mount("/assets", StaticFiles(directory=dashboard_dir), name="assets")

        @app.get("/")
        async def serve_dashboard():
            index_path = os.path.join(dashboard_dir, "index.html")
            if os.path.isfile(index_path):
                return FileResponse(index_path)
            return {"message": "Flight Recorder API — dashboard not found"}

    return app
